import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from interview.controller import InterviewController
from interview.report import generate_report

def main():
    controller = InterviewController()
    print("Starting session...")
    controller.start_session('data/raw/resume.txt', 'data/raw/skills.txt')
    
    print("Topics selected:", controller.state.topics_to_cover)
    
    for _ in range(5):
        q = controller.next_question()
        if not q: break
        print("Q:", q)
        ans = "I think python is very good and I use lists."
        score = controller.process_answer(ans)
        print("Score:", score)
        
    path, content = generate_report(controller.state)
    print("Report generated at:", path)
    print(content)

if __name__ == "__main__":
    main()
