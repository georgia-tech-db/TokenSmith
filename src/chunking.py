import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional
import backoff
from tqdm import tqdm

from nltk import sent_tokenize
from transformers import AutoTokenizer

from langchain_openai import ChatOpenAI
from pydantic import BaseModel
from langchain.chains import create_extraction_chain_pydantic
from langchain import hub


# -------------------------- Chunking Configs --------------------------
class ChunkConfig(ABC):
    def validate(self):
        pass

    def to_string(self):
        return ""

@dataclass
class CharChunkConfig(ChunkConfig):
    max_chars: int

    def to_string(self):
        return f"chunk_mode=chars, max_chars={self.max_chars}"

    def validate(self):
        assert self.max_chars > 0, "chunk_size_char must be > 0"

@dataclass
class TokenChunkConfig(ChunkConfig):
    max_tokens: int

    def to_string(self):
        return f"chunk_mode=tokens, max_tokens={self.max_tokens}"

    def validate(self):
        assert self.max_tokens > 0, "chunk_size_char must be > 0"

@dataclass
class SlidingTokenConfig(ChunkConfig):
    max_tokens: int
    overlap_tokens: int
    tokenizer_name: str

    def to_string(self):
        return (
            f"chunk_mode=sliding-tokens, "
            f"max_tokens={self.max_tokens}, "
            f"overlap_tokens={self.overlap_tokens}, "
            f"tokenizer_name={self.tokenizer_name}"
        )

    def validate(self):
        assert self.max_tokens > 0, "chunk_size_char must be > 0"
        assert self.overlap_tokens > 0, "chunk_size_char must be > 0"

@dataclass
class SectionChunkConfig(ChunkConfig):
    def to_string(self):
        return "chunk_mode=section"

@dataclass
class LLMChunkConfig(ChunkConfig):
    max_tokens: int
    overlap_tokens: int
    tokenizer_name: str
    open_ai_api_key: str
    
    def to_string(self):
        return (
            f"chunk_mode=llm, "
            f"max_tokens={self.max_tokens}, "
            f"overlap_tokens={self.overlap_tokens}, "
            f"tokenizer_name={self.tokenizer_name}, "
            f"open_ai_api_key={self.open_ai_api_key}, "
        )

    def validate(self):
        assert self.max_tokens > 0, "chunk_size_char must be > 0"
        assert self.overlap_tokens > 0, "chunk_size_char must be > 0"

@dataclass
class PropositionalChunkConfig(ChunkConfig):
    model_name: str = "gpt-3.5-turbo"
    hub_spec: str = "wfh/proposal-indexing"
    open_ai_api_key: str = ""
    max_paragraphs: Optional[int] = None   # safety limit

    def to_string(self):
        return f"chunk_mode=propositional, model={self.model_name}, hub={self.hub_spec}"

    def validate(self):
        assert isinstance(self.model_name, str) and self.model_name, "model_name must be a non-empty string"
        assert isinstance(self.hub_spec, str) and self.hub_spec, "hub_spec must be a non-empty string"

# -------------------------- Chunking Strategies --------------------------

class ChunkStrategy(ABC):
    """Abstract base for all chunking strategies."""
    @abstractmethod
    def name(self) -> str: ...
    @abstractmethod
    def chunk(self, text: str) -> List[str]: ...
    @abstractmethod
    def artifact_folder_name(self) -> str: ...

class CharChunkStrategy(ChunkStrategy):
    def __init__(self, config: CharChunkConfig):
        self.max_chars = int(config.max_chars)

    def name(self) -> str:
        return f"chars-{self.max_chars}"

    def artifact_folder_name(self) -> str:
        return f"chars({self.max_chars})"

    def chunk(self, text: str) -> List[str]:
        step = max(1, self.max_chars)
        return [text[i:i + step] for i in range(0, len(text), step)]

class SentencePackStrategy(ChunkStrategy):
    """
    Sentence-aware packing using word-count as a token proxy.
    """
    def __init__(self, config: TokenChunkConfig):
        self.max_tokens = int(config.max_tokens)

    def name(self) -> str:
        return f"tokens(word-proxy:{self.max_tokens})"

    def artifact_folder_name(self) -> str:
        return f"tokens-{self.max_tokens}"

    def chunk(self, text: str) -> List[str]:
        chunks: List[str] = []
        cur, cur_len = [], 0
        for s in sent_tokenize(text):
            w = len(s.split())
            if cur and cur_len + w > self.max_tokens:
                chunks.append(" ".join(cur))
                cur, cur_len = [s], w
            else:
                cur.append(s)
                cur_len += w
        if cur:
            chunks.append(" ".join(cur))
        return chunks

class SlidingTokenStrategy(ChunkStrategy):
    """
    True token windows with overlap using HF tokenizer.
    window = max_tokens, step = max_tokens - overlap_tokens
    """
    def __init__(self, config: SlidingTokenConfig):
        self.tokenizer_name = config.tokenizer_name
        self.max_tokens = int(config.max_tokens)
        self.overlap_tokens = int(config.overlap_tokens)
        self._tok = AutoTokenizer.from_pretrained(config.tokenizer_name,
                                                  use_fast=True,
                                                  model_max_length=1_000_000_000)

    def name(self) -> str:
        return f"sliding-tokens({self.max_tokens},{self.overlap_tokens})"

    def artifact_folder_name(self) -> str:
        return f"sliding-tokens-{self.max_tokens}-{self.overlap_tokens}"

    def chunk(self, text: str) -> List[str]:
        ids = self._tok.encode(text, add_special_tokens=False)
        if not ids or self.max_tokens <= 0:
            return []
        step = self.max_tokens - self.overlap_tokens
        if step <= 0:
            step = self.max_tokens  # safety

        chunks: List[str] = []
        for i in range(0, len(ids), step):
            window = ids[i:i + self.max_tokens]
            if not window:
                break
            chunk_text = self._tok.decode(
                window,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            chunks.append(chunk_text)
        return chunks

class SectionStrategy(ChunkStrategy):
    """
    Splits text into chunks based on numeric section headings.
    Example matches:
      1. Introduction
      2.3 Subtopic
      10.4.1 Deep Dive
    Collects text until the next heading of same-or-higher level.
    """
    HEADING_RE = re.compile(
        r"""
        (?m)                               # multiline
        ^(?=.{,120}$)\s{0,3}               # shortish heading line
        (?P<num>[1-9]\d*(?:\.[0-9]+)+)     # 1.2 or 10.4.1 etc.
        (?![)\]])                          # avoid "1.2)"
        \s+(?P<title>(?!\d).+?)\s*$        # title text
        """,
        re.VERBOSE,
    )

    def name(self) -> str:
        return "sections"

    def artifact_folder_name(self) -> str:
        return "sections"

    def chunk(self, text: str) -> List[str]:
        matches = list(self.HEADING_RE.finditer(text))
        if not matches:
            # No headings detected → return the whole text as one chunk
            return [text.strip()] if text.strip() else []

        heads = []
        for m in matches:
            num = m.group("num")
            title = m.group("title").strip()
            level = num.count(".") + 1
            heads.append({
                "num": num, "title": title, "level": level,
                "start": m.start(), "endline": m.end()
            })

        chunks: List[str] = []
        N = len(heads)
        for i, h in enumerate(heads):
            end_idx = len(text)
            for j in range(i + 1, N):
                if heads[j]["level"] <= h["level"]:
                    end_idx = heads[j]["start"]
                    break
            body = text[h["endline"]:end_idx].strip("\n").strip()
            if body:
                chunks.append(body)

        # If there's preface text before the first heading, keep it as a chunk
        preface_start = 0
        first_start = heads[0]["start"]
        preface = text[preface_start:first_start].strip()
        if preface:
            chunks.insert(0, preface)

        return chunks

# -------------------------- LLM Chunking Strategy --------------------------

class LLMSemanticChunker(ChunkStrategy):
    """
    LLMSemanticChunker uses OpenAI's API to split text into thematically consistent sections.
    """
    def __init__(self, config: LLMChunkConfig):
        self.config = config

        self.model_name = "gpt-4o"
        self.response_max_tokens = 200
        self.temperature = 0.2

        from openai import OpenAI
        import os
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY", ""))

        # Set up initial chunker for preprocessing
        self.splitter = SlidingTokenStrategy(SlidingTokenConfig(
            max_tokens=config.max_tokens,
            overlap_tokens=config.overlap_tokens,
            tokenizer_name=config.tokenizer_name
        ))

    def name(self) -> str:
        return f"llm-semantic(openai:{self.config.model_name})"

    def artifact_folder_name(self) -> str:
        return f"llm-semantic-openai-{self.model_name}"

    @backoff.on_exception(backoff.expo, Exception, max_tries=3)
    def _create_message(self, system_prompt, messages, max_tokens=1000, temperature=1.0):
        """Create a message using OpenAI API with retry logic."""
        try:
            gpt_messages = [
                {"role": "system", "content": system_prompt}
            ] + messages

            completion = self.openai_client.chat.completions.create(
                model=self.model_name,
                max_tokens=max_tokens,
                messages=gpt_messages,
                temperature=temperature
            )

            return completion.choices[0].message.content
        except Exception as e:
            print(f"Error occurred: {e}, retrying...")
            raise e

    def get_prompt(self, chunked_input, current_chunk=0, invalid_response=None):
        messages = [
            {
                "role": "system", 
                "content": (
                    "You are an assistant specialized in splitting text into thematically consistent sections. "
                    "The text has been divided into chunks, each marked with <|start_chunk_X|> and <|end_chunk_X|> tags, where X is the chunk number. "
                    "Your task is to identify the points where splits should occur, such that consecutive chunks of similar themes stay together. "
                    "Respond with a list of chunk IDs where you believe a split should be made. For example, if chunks 1 and 2 belong together but chunk 3 starts a new topic, you would suggest a split after chunk 2. THE CHUNKS MUST BE IN ASCENDING ORDER."
                    "Your response should be in the form: 'split_after: 3, 5'."
                )
            },
            {
                "role": "user", 
                "content": (
                    "CHUNKED_TEXT: " + chunked_input + "\n\n"
                    "Respond only with the IDs of the chunks where you believe a split should occur. YOU MUST RESPOND WITH AT LEAST ONE SPLIT. THESE SPLITS MUST BE IN ASCENDING ORDER AND EQUAL OR LARGER THAN: " + str(current_chunk)+"." + (f"\n The previous response of {invalid_response} was invalid. DO NOT REPEAT THIS ARRAY OF NUMBERS. Please try again." if invalid_response else "")
                )
            },
        ]
        return messages

    def openai_token_count(self, text):
        """Simple token count approximation using word count * 1.3"""
        return int(len(text.split()) * 1.3)

    def _build_chunked_input(self, chunks: List[str], current_chunk: int) -> str:
        """Build input string for LLM with chunk markers."""
        token_count = 0
        chunked_input = ''
        
        for i in range(current_chunk, len(chunks)):
            token_count += self.openai_token_count(chunks[i])
            chunked_input += f"<|start_chunk_{i+1}|>{chunks[i]}<|end_chunk_{i+1}|>"
            if token_count > 800:
                break
                
        return chunked_input

    def _get_llm_response(self, chunked_input: str, current_chunk: int) -> List[int]:
        """Get LLM response with retry logic for split points."""
        messages = self.get_prompt(chunked_input, current_chunk)
        
        while True:
            result_string = self._create_message(
                messages[0]['content'], 
                messages[1:], 
                max_tokens=self.response_max_tokens, 
                temperature=self.temperature
            )
            
            # Parse response to extract split points
            split_after_line = [line for line in result_string.split('\n') if 'split_after:' in line]
            if not split_after_line:
                print("No valid split_after line found, retrying...")
                messages = self.get_prompt(chunked_input, current_chunk, "no split_after line")
                continue
                
            numbers = re.findall(r'\d+', split_after_line[0])
            numbers = list(map(int, numbers))

            # Validate response
            if numbers and numbers == sorted(numbers) and all(number >= current_chunk for number in numbers):
                return numbers
            else:
                messages = self.get_prompt(chunked_input, current_chunk, numbers)
                print("Response: ", result_string)
                print("Invalid response. Please try again.")

    def _reconstruct_chunks(self, chunks: List[str], split_indices: List[int]) -> List[str]:
        """Reconstruct final chunks based on split indices."""
        chunks_to_split_after = [i - 1 for i in split_indices]
        
        docs = []
        current_chunk = ''
        for i, chunk in enumerate(chunks):
            current_chunk += chunk + ' '
            if i in chunks_to_split_after:
                docs.append(current_chunk.strip())
                current_chunk = ''
        if current_chunk:
            docs.append(current_chunk.strip())
            
        return docs

    def chunk(self, text: str) -> List[str]:
        """Main chunking method using LLM for semantic splitting."""
        # First, create initial chunks using sliding token strategy
        chunks = self.splitter.chunk(text)
        
        if len(chunks) <= 1:
            return chunks

        split_indices = []
        current_chunk = 0

        with tqdm(total=len(chunks), desc="Processing chunks with LLM") as pbar:
            while True:
                if current_chunk >= len(chunks) - 4:
                    break

                chunked_input = self._build_chunked_input(chunks, current_chunk)
                numbers = self._get_llm_response(chunked_input, current_chunk)

                split_indices.extend(numbers)
                current_chunk = numbers[-1] if numbers else len(chunks)

                if not numbers:
                    break

                pbar.update(current_chunk - pbar.n)

        pbar.close()

        return self._reconstruct_chunks(chunks, split_indices)

# -------------------------- LLM Proposition Chunkign Strategy -----------------------------

class Sentences(BaseModel):
    sentences: List[str]

class PropositionalChunkStrategy(ChunkStrategy):
    """
    Uses an LLM extraction chain to break text into propositions (atomic sentences).
    """

    def __init__(self, config: PropositionalChunkConfig):
        config.validate()
        self.config = config

        # LLM + Hub prompt
        self.llm = ChatOpenAI(model=config.model_name, api_key=config.open_ai_api_key)
        self.prompt = hub.pull(config.hub_spec)
        self.runnable = self.prompt | self.llm

        # Schema-driven extraction
        self.extraction_chain = create_extraction_chain_pydantic(
            pydantic_schema=Sentences,
            llm=self.llm
        )

    def name(self) -> str:
        return f"llm-propositional({self.config.model_name})"

    def artifact_folder_name(self) -> str:
        return f"llm-propositional-{self.config.model_name.replace('.', '-')}"
    
    def _get_propositions(self, text: str) -> List[str]:
        """Run LLM pipeline to extract propositions from a single paragraph."""
        runnable_output = self.runnable.invoke({"input": text}).content
        try:
            propositions = self.extraction_chain.invoke(runnable_output)["text"][0].sentences
        except Exception:
            propositions = []
        return propositions

    def chunk(self, text: str) -> List[str]:
        """Split text into paragraphs, extract propositions from each."""
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if self.config.max_paragraphs:
            paragraphs = paragraphs[: self.config.max_paragraphs]

        chunks: List[str] = []
        for i, para in enumerate(paragraphs):
            props = self._get_propositions(para)
            chunks.extend(props)
            print(f"✓ Paragraph {i+1}/{len(paragraphs)} → {len(props)} propositions")

        return chunks

# -------------------------- Strategy Factory -----------------------------

def make_chunk_strategy(
    config: ChunkConfig
) -> ChunkStrategy:
    if isinstance(config, CharChunkConfig):
        return CharChunkStrategy(config)
    if isinstance(config, TokenChunkConfig):
        return SentencePackStrategy(config)
    if isinstance(config, SlidingTokenConfig):
        return SlidingTokenStrategy(config)
    if isinstance(config, SectionChunkConfig):
        return SectionStrategy()
    if isinstance(config, LLMChunkConfig):
        return LLMSemanticChunker(config)
    if isinstance(config, PropositionalChunkConfig):
        return PropositionalChunkStrategy(config)
    raise ValueError(f"Unknown chunk config type: {config.__class__.__name__}")
