import os
from datetime import datetime
from config import REPORTS_DIR, STRONG_SCORE_THRESHOLD, WEAK_SCORE_THRESHOLD, PROCTOR_VIOLATION_LIMIT

def generate_report(state):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    filename  = f"session_{timestamp}.txt"
    filepath  = os.path.join(REPORTS_DIR, filename)

    total_score = 0
    total_q     = 0

    report_lines = []
    report_lines.append(f"AI Mock Interview Report - {timestamp}")
    report_lines.append("=" * 40)

    for topic, scores in state.scores.items():
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        total_score += sum(scores)
        total_q     += len(scores)

        report_lines.append(f"\nTopic: {topic.replace('_', ' ').title()}")
        report_lines.append(f"Questions Asked: {len(scores)}")
        report_lines.append(f"Average Score: {avg:.2f}")

        weak_count   = sum(1 for s in scores if s < WEAK_SCORE_THRESHOLD)
        strong_count = sum(1 for s in scores if s >= STRONG_SCORE_THRESHOLD)

        report_lines.append(f"Strong Answers: {strong_count}")
        report_lines.append(f"Weak Answers: {weak_count}")

    report_lines.append("\n" + "=" * 40)

    overall_avg = total_score / total_q if total_q > 0 else 0
    report_lines.append(f"Overall Average Score: {overall_avg:.2f}")

    if overall_avg < 0.5:
        level = "Junior-Ready"
    elif overall_avg < 0.7:
        level = "Mid-Level"
    else:
        level = "Senior-Ready"

    report_lines.append(f"Readiness Level: {level}")

    # ── Exam Proctor Integrity Note ───────────────────────────────────────────
    violations = getattr(state, 'proctor_violations', 0)
    proctor_score = getattr(state, 'proctor_score', 10)
    
    report_lines.append("\n" + "=" * 40)
    report_lines.append("🛡️  EXAM INTEGRITY & AI PROCTORING REPORT")
    report_lines.append("=" * 40)
    report_lines.append(f"Final Proctor Score: {proctor_score}/10")
    report_lines.append(f"Total Attention / Gaze / Object Violations: {violations}")
    
    breakdown = getattr(state, 'proctor_violations_breakdown', {})
    if breakdown:
        report_lines.append("\nViolation Breakdown:")
        report_lines.append(f"- No face: {breakdown.get('no_face', 0)}")
        report_lines.append(f"- Look away: {breakdown.get('look_away', 0)}")
        report_lines.append(f"- Multiple people: {breakdown.get('multiple_people', 0)}")
        report_lines.append(f"- Phone detected: {breakdown.get('phone_detected', 0)}")
        report_lines.append(f"- Different person: {breakdown.get('different_person', 0)}")
        
    logs = getattr(state, 'proctor_violations_log', [])
    if logs:
        report_lines.append("\nDetailed Integrity Log:")
        for log in logs:
            screenshot = log.get('screenshot_path', '')
            screenshot_str = f"\n  ![Violation Screenshot]({screenshot})" if screenshot else ""
            report_lines.append(f"- [{log.get('timestamp', '')}] {log.get('type', '').upper()}: {log.get('message', '')}{screenshot_str}")
    else:
        report_lines.append("\nDetailed Integrity Log:")
        report_lines.append("- No attention or device violations recorded. Gaze integrity perfectly maintained!")

    if violations >= PROCTOR_VIOLATION_LIMIT:
        report_lines.append("\n" + "=" * 40)
        report_lines.append(
            f"⚠️  Integrity Warning: Candidate attention violations exceeded the limits "
            f"during this session ({violations} violation(s) detected). Session flagged for manual review."
        )

    report_content = "\n".join(report_lines)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(report_content)

    return filepath, report_content

