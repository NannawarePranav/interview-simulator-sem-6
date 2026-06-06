import os
import sys
import time
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.progress import Progress, BarColumn, TextColumn
from rich import print as rprint

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from interview.controller import InterviewController
from interview.report import generate_report
from config import ANSWER_TIMEOUT_SECONDS

console = Console()

def get_input_timeout_win(prompt, timeout=90):
    import msvcrt
    console.print(prompt, end='')
    start_time = time.time()
    input_str = ''
    while True:
        if msvcrt.kbhit():
            # read character
            char = msvcrt.getwch()
            if char in ('\r', '\n'):
                console.print()
                return input_str
            elif char == '\b':
                if len(input_str) > 0:
                    input_str = input_str[:-1]
                    sys.stdout.write('\b \b')
                    sys.stdout.flush()
            elif char == '\x03': # ctrl+c
                raise KeyboardInterrupt
            else:
                input_str += char
                sys.stdout.write(char)
                sys.stdout.flush()
        if time.time() - start_time > timeout:
            console.print("\n[red]Timeout![/red]")
            return None
        time.sleep(0.01)

def display_progress(state):
    # Live progress line: Topics: [Python ✓] [DSA ...] [Behavioral —]
    all_topics = state.topics_covered + state.topics_to_cover
    if state.current_topic and state.current_topic not in all_topics:
        all_topics.insert(len(state.topics_covered), state.current_topic)
        
    topic_str = "Topics: "
    for t in all_topics:
        name = t.replace('_', ' ').title()
        if t in state.topics_covered:
            topic_str += f"[green][{name} ✓][/green] "
        elif t == state.current_topic:
            topic_str += f"[bold yellow][{name} ...][/bold yellow] "
        else:
            topic_str += f"[dim][{name} —][/dim] "
            
    console.print(Panel(topic_str, title="Live Progress"))

def main():
    parser = argparse.ArgumentParser(description="AI Mock Interviewer - V1")
    parser.add_argument("--resume", required=True, help="Path to resume.txt")
    parser.add_argument("--skills", required=True, help="Path to skills.txt")
    args = parser.parse_args()
    
    console.print(Panel("[bold cyan]Welcome to AI Mock Interviewer - V1[/bold cyan]", expand=False))
    
    controller = InterviewController()
    console.print("[dim]Initializing session and models...[/dim]")
    controller.start_session(args.resume, args.skills)
    
    total_q_count = 0
    total_score = 0.0
    
    try:
        while True:
            display_progress(controller.state)
            question = controller.next_question()
            
            if not question:
                console.print("\n[bold green]Interview Completed![/bold green]")
                break
                
            total_q_count += 1
            curr_avg = total_score / (total_q_count - 1) if total_q_count > 1 else 0.0
            
            console.print(f"\n[bold magenta]Q{total_q_count} ({controller.state.current_topic}):[/bold magenta] {question}")
            console.print(f"[dim]Current Avg Score: {curr_avg:.2f} | Type your answer (you have {ANSWER_TIMEOUT_SECONDS}s). Use \[skip] or \[quit][/dim]")
            
            ans = get_input_timeout_win("> ", timeout=ANSWER_TIMEOUT_SECONDS)
            
            if ans is None: # timeout
                console.print("[yellow]Time is up. Score penalized.[/yellow]")
                score = 0.1
                controller.state.scores[controller.state.current_topic].append(score)
                total_score += score
            elif ans.strip().lower() == '[quit]':
                console.print("[yellow]Ending session early.[/yellow]")
                break
            elif ans.strip().lower() == '[skip]':
                console.print("[yellow]Question skipped. Score penalized.[/yellow]")
                score = 0.1
                controller.state.scores[controller.state.current_topic].append(score)
                total_score += score
            else:
                score = controller.process_answer(ans)
                total_score += score
                
                # Immediate feedback
                if score >= 0.7:
                    console.print(f"[{score:.2f}] [green]Strong Answer![/green]")
                elif score >= 0.4:
                    console.print(f"[{score:.2f}] [yellow]Mediocre Answer.[/yellow]")
                else:
                    console.print(f"[{score:.2f}] [red]Weak Answer.[/red]")
                    
    except KeyboardInterrupt:
        console.print("\n[red]Interview aborted by user.[/red]")
        
    console.print("\n[bold cyan]Generating Report...[/bold cyan]")
    path, content = generate_report(controller.state)
    console.print(Panel(content, title=f"Saved to {os.path.basename(path)}"))
    
if __name__ == "__main__":
    main()
