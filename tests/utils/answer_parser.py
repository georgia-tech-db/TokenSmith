import re

ANSWER_START = "=== ANSWER ========================================="
ANSWER_END = "===================================================="

def extract_answer_from_output(output):
    """Extract the answer from TokenSmith chat output."""
    lines = output.strip().split('\n')
    
    # Look for answer section between markers
    answer_lines = []
    in_answer_section = False
    
    for line in lines:
        # Start capturing after "=== ANSWER ===" marker
        if ANSWER_START in line:
            in_answer_section = True
            continue
        
        # Stop capturing at end marker or new prompt
        if in_answer_section and (ANSWER_END in line or "Ask >" in line or line.strip() == ""):
            if ANSWER_END in line:
                break
            if line.strip() == "":
                continue
        
        if in_answer_section and line.strip():
            answer_lines.append(line.strip())
    
    # if not answer_lines:
    #     # Look for content after the last "Ask >" prompt
    #     for i, line in enumerate(reversed(lines)):
    #         if "Ask >" in line:
    #             # Take the lines after this prompt (before it in reversed order)
    #             answer_lines = [l.strip() for l in lines[len(lines)-i:] 
    #                            if l.strip() and not l.startswith("Ask >")]
    #             break
    
    # Clean up the answer
    answer = ' '.join(answer_lines)
    answer = re.sub(r'\(no output\)', '', answer)
    answer = re.sub(r'\s+', ' ', answer)
    answer = answer.strip()
    
    return answer if answer else "No answer found"
