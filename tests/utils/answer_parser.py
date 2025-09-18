import re

ANSWER_START = "=== ANSWER ========================================="
ANSWER_END = "===================================================="

def extract_answer_from_output(output):
    """Extract the answer from TokenSmith chat output."""
    lines = output.strip().split('\n')
    
    # Look for answer section between markers
    answer_lines = []
    in_answer_section = False
    llama_perf_pattern = re.compile(r'^llama_perf_.*$')
    
    for line in lines:
        if ANSWER_START in line:
            in_answer_section = True
            continue
        
        if in_answer_section and (ANSWER_END in line or "Ask >" in line or line.strip() == ""):
            if ANSWER_END in line:
                break
            if line.strip() == "":
                continue
        
        if in_answer_section and line.strip():
            if llama_perf_pattern.match(line.strip()):
                continue
            answer_lines.append(line.strip())
    
    answer = ' '.join(answer_lines)
    answer = re.sub(r'\(no output\)', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    return answer if answer else "No answer found"
