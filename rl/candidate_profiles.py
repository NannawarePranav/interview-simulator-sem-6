import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TOPICS

CANDIDATE_PROFILES = {
    "strong_python_weak_sql": {
        "python": 0.9, 
        "data_structures_algorithms": 0.6, 
        "machine_learning_basics": 0.4, 
        "databases_sql": 0.2, 
        "system_design": 0.6, 
        "behavioral": 0.8
    },
    "junior_all_around": {topic: 0.35 for topic in TOPICS},
    "senior_system_design": {
        "python": 0.7, 
        "data_structures_algorithms": 0.8, 
        "machine_learning_basics": 0.5, 
        "databases_sql": 0.8, 
        "system_design": 0.95, 
        "behavioral": 0.9
    },
    "ml_specialist": {
        "python": 0.85, 
        "data_structures_algorithms": 0.5, 
        "machine_learning_basics": 0.95, 
        "databases_sql": 0.6, 
        "system_design": 0.4, 
        "behavioral": 0.7
    },
    "well_rounded_mid": {topic: 0.6 for topic in TOPICS},
}
