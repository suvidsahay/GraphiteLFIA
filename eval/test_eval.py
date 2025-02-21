from .combined_eval import CombinedEvaluator

instruction = "Generate an article about military careers."
response = "Exploring Military Careers: Opportunities, Benefits, and Challenges.\
A career in the military offers a unique path that combines service, discipline, and numerous opportunities for personal and professional growth. Whether joining as an enlisted service member or an officer, the military provides a structured environment with diverse career options, educational benefits, and long-term stability. This article explores the key aspects of a military career, including different roles, advantages, and potential challenges. \
Types of Military Careers \
Military careers can be broadly categorized into enlisted and officer roles: \
Enlisted Personnel: These individuals carry out the daily operations of the military, serving in roles such as infantry, mechanics, cybersecurity specialists, and medical technicians. Enlisted members typically receive training through boot camp and specialized technical schools. \
Officers: Officers hold leadership positions and are responsible for strategy, planning, and command. They may enter the military through service academies, ROTC programs, or Officer Candidate School (OCS). Officers typically serve in fields like engineering, intelligence, logistics, and aviation. \
Benefits of a Military Career \
A career in the armed forces offers many advantages, including: \
Job Security & Stable Income: Military personnel receive competitive salaries, with additional allowances for housing, food, and special duties. \
Education & Training: Many branches offer tuition assistance, the GI Bill for higher education, and extensive on-the-job training. \
Health & Retirement Benefits: Service members and their families receive comprehensive healthcare and retirement plans, including pensions for those who serve 20 years or more. \
Skill Development & Career Advancement: Military service fosters discipline, leadership, teamwork, and technical expertise that can be valuable in both military and civilian careers. \
Travel & Adventure: Service members may be stationed in various locations worldwide, experiencing different cultures and environments. \
Challenges of Military Service \
Despite its benefits, a military career also comes with challenges: \
Physical & Mental Demands: Service members must maintain high physical fitness standards and may face stress from combat or deployments. \
Family & Lifestyle Adjustments: Frequent relocations and long deployments can impact family life and personal relationships. \
Risk & Uncertainty: Military personnel may be placed in dangerous situations, requiring them to be prepared for combat and other emergencies. \
Structured Environment: The military operates under strict rules and regulations, which may not suit everyone's personality or career aspirations. \
Choosing the Right Path \
Before joining the military, it is important to research different branches (Army, Navy, Air Force, Marine Corps, Coast Guard, and Space Force) and their specific roles. Speaking with recruiters, current service members, or veterans can provide valuable insights into military life. \
A military career can be a rewarding experience, offering unique opportunities for growth, service, and leadership. However, it requires commitment, adaptability, and resilience. By understanding the benefits and challenges, individuals can make informed decisions about whether a military path aligns with their personal and professional goals."

reference = "The military provides diverse career paths, structured training, and global opportunities..."
evaluator = CombinedEvaluator()
results = evaluator.evaluate_all(instruction, response, reference)

def main():
    for category, result in results.items():
        print(f"{category}: {result}")
