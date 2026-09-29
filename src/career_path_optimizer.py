"""
Career Path Optimizer Module
Uses Hill Climbing and A* Search algorithms to find the optimal skill learning path.

AI Concepts Demonstrated:
    - State Space Search: States = skill sets, Operators = learning a skill
    - Hill Climbing: Greedy local search, picks the best immediate improvement
    - A* Algorithm: Optimal search using g(n) + h(n) with admissible heuristic
"""

import heapq
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Set, Tuple, Optional


# ─── Skill Knowledge Base ─────────────────────────────────────────────────────
# Each skill: category, estimated learning days, prerequisites, description

def load_skill_database():
    """Load the skills dataset from JSON file."""
    json_path = os.path.join(os.path.dirname(__file__), "skills_dataset.json")
    try:
        with open(json_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Fallback empty DB if file missing
        return {}

SKILL_DATABASE = load_skill_database()




def normalize_skill(skill: str) -> str:
    """Normalize a skill name for matching."""
    return skill.lower().strip()


def extract_skills_from_text(text: str) -> Set[str]:
    """
    Extract known skills from text by matching against the skill database.
    Uses word-boundary aware matching to reduce false positives.
    """
    import re
    text_lower = text.lower()
    found_skills = set()

    # Sort skills by length (longest first) to match multi-word skills before single-word ones
    sorted_skills = sorted(SKILL_DATABASE.keys(), key=len, reverse=True)

    for skill_name in sorted_skills:
        # Use word boundary matching for single-word skills
        # For multi-word skills, check direct substring presence
        pattern = r'\b' + re.escape(skill_name) + r'\b'
        if re.search(pattern, text_lower):
            found_skills.add(skill_name)

    return found_skills


def get_all_required_skills(target_skills: Set[str], current_skills: Set[str]) -> Set[str]:
    """
    Given target skills, find all skills needed including prerequisites
    that the candidate doesn't already have.
    """
    all_needed = set(target_skills)
    queue = list(target_skills)
    while queue:
        skill = queue.pop()
        if skill in SKILL_DATABASE:
            for prereq in SKILL_DATABASE[skill]["prerequisites"]:
                if prereq not in current_skills and prereq not in all_needed:
                    all_needed.add(prereq)
                    queue.append(prereq)
    return all_needed


def get_learnable_skills(current_skills: Set[str], all_needed: Set[str]) -> List[str]:
    """Get skills that can be learned next (all prerequisites already acquired)."""
    missing = all_needed - current_skills
    learnable = []

    for skill in missing:
        if skill in SKILL_DATABASE:
            prereqs = set(SKILL_DATABASE[skill]["prerequisites"])
            if prereqs.issubset(current_skills):
                learnable.append(skill)

    return learnable


def calculate_match_score(current_skills: Set[str], target_skills: Set[str]) -> float:
    """Calculate match percentage between current and target skills."""
    if not target_skills:
        return 100.0
    matched = current_skills.intersection(target_skills)
    return (len(matched) / len(target_skills)) * 100


# ─── Hill Climbing Algorithm ──────────────────────────────────────────────────

def hill_climbing_optimizer(current_skills: Set[str], target_skills: Set[str]) -> dict:
    """
    Use Hill Climbing to find a skill learning path.

    Algorithm:
        1. Start with the candidate's current skill set (initial state)
        2. Evaluate all learnable skills (whose prerequisites are satisfied)
        3. Pick the skill with the BEST value = (match_improvement / effort_hours)
        4. Add it to the skill set and repeat
        5. Stop when goal is reached or no improvement is possible (local maximum)

    This is a GREEDY approach — it may not find the globally optimal path,
    but it's fast and produces reasonable results.

    Returns:
        Dictionary with path details, scores, and algorithm execution trace
    """
    # Expand target to include prerequisites
    all_needed = get_all_required_skills(target_skills, current_skills)

    path = []
    acquired = set(current_skills)
    initial_score = calculate_match_score(acquired, target_skills)
    trace = []
    step = 0

    trace.append({
        "step": step,
        "action": "START",
        "current_score": round(initial_score, 1),
        "skills_known": len(acquired),
        "skills_needed": len(all_needed - acquired),
        "description": f"Initial state: {round(initial_score, 1)}% match | "
                       f"{len(acquired & target_skills)}/{len(target_skills)} target skills known"
    })

    while not all_needed.issubset(acquired):
        step += 1
        learnable = get_learnable_skills(acquired, all_needed)

        if not learnable:
            trace.append({
                "step": step,
                "action": "STUCK — LOCAL MAXIMUM",
                "current_score": round(calculate_match_score(acquired, target_skills), 1),
                "description": "Hill Climbing stuck! No learnable skills with satisfied prerequisites. "
                               "This is a limitation of greedy search — it cannot backtrack."
            })
            break

        # ── Evaluate all neighbors (learnable skills) ──
        best_skill = None
        best_value = -1
        evaluations = []

        for skill in learnable:
            new_skills = acquired | {skill}
            new_score = calculate_match_score(new_skills, target_skills)
            old_score = calculate_match_score(acquired, target_skills)
            improvement = new_score - old_score
            effort = SKILL_DATABASE[skill].get("effort_days", 10)
            # Value function: score improvement per day of effort
            # Skills that are prerequisites (not in target) get 0 improvement
            # but still contribute by unlocking other skills
            value = (improvement / effort) if effort > 0 else improvement
            # Bonus for prerequisite skills (they unlock more skills)
            if skill not in target_skills:
                value += 0.001  # Small bonus to prefer prereqs when tied

            evaluations.append({
                "skill": skill,
                "category": SKILL_DATABASE[skill]["category"],
                "improvement": round(improvement, 2),
                "effort_days": effort,
                "value_score": round(value, 6)
            })

            if value > best_value:
                best_value = value
                best_skill = skill

        if best_skill is None:
            break

        # ── Take the uphill step ──
        acquired.add(best_skill)
        new_score = calculate_match_score(acquired, target_skills)
        skill_info = SKILL_DATABASE[best_skill]

        path.append({
            "step": step,
            "skill": best_skill,
            "category": skill_info["category"],
            "effort_days": skill_info.get("effort_days", 10),
            "score_after": round(new_score, 1),
            "is_prerequisite": best_skill not in target_skills,
            "description": skill_info.get("description", "")
        })

        trace.append({
            "step": step,
            "action": f"CLIMB → Learn '{best_skill}'",
            "evaluations_count": len(evaluations),
            "chosen_skill": best_skill,
            "chosen_value": round(best_value, 6),
            "current_score": round(new_score, 1),
            "is_prerequisite": best_skill not in target_skills,
            "description": f"Evaluated {len(evaluations)} neighbors | "
                           f"Best: '{best_skill}' (value={round(best_value, 6)}) | "
                           f"Score: {round(new_score, 1)}%"
        })

    final_score = calculate_match_score(acquired, target_skills)
    total_effort = sum(s["effort_days"] for s in path)

    return {
        "algorithm": "Hill Climbing (Greedy Local Search)",
        "initial_score": round(initial_score, 1),
        "final_score": round(final_score, 1),
        "path": path,
        "total_effort_days": total_effort,
        "total_skills_to_learn": len(path),
        "trace": trace,
        "success": final_score >= 100.0
    }


# ─── A* Algorithm ─────────────────────────────────────────────────────────────

@dataclass(order=True)
class AStarNode:
    """Node for A* search priority queue."""
    f_score: float
    g_score: float = field(compare=False)
    skills: frozenset = field(compare=False)
    path: tuple = field(compare=False)


def heuristic(current_skills: frozenset, all_needed: Set[str]) -> float:
    """
    Admissible heuristic for A* search.

    Estimates the remaining cost as the sum of effort for all missing skills.
    This is ADMISSIBLE because in reality, the actual cost is at least this much
    (we might also need prerequisites not counted here, making actual cost higher).

    h(n) = Σ effort(skill) for each skill in (all_needed - current_skills)
    """
    missing = all_needed - current_skills
    h = 0
    for skill in missing:
        if skill in SKILL_DATABASE:
            h += SKILL_DATABASE[skill].get("effort_days", 10)
        else:
            h += 7  # Default estimate in days for unknown skills
    return h


def a_star_optimizer(current_skills: Set[str], target_skills: Set[str],
                     max_iterations: int = 10000) -> dict:
    """
    Use A* algorithm to find the OPTIMAL (minimum total effort) skill learning path.

    Algorithm:
        - State: frozenset of skills the candidate currently knows
        - g(n): total effort hours spent so far
        - h(n): admissible heuristic — sum of effort for remaining missing skills
        - f(n) = g(n) + h(n): estimated total cost through this node
        - Expand: try learning each skill whose prerequisites are satisfied
        - Goal: all target skills (and their prerequisites) are acquired

    A* guarantees the OPTIMAL path when the heuristic is admissible (never overestimates).

    Returns:
        Dictionary with optimal path details, nodes explored, and algorithm trace
    """
    initial_score = calculate_match_score(current_skills, target_skills)

    # Find all skills needed (targets + prerequisites)
    all_needed = get_all_required_skills(target_skills, current_skills)
    # Only consider skills we actually need to learn
    skills_to_learn = all_needed - current_skills

    if not skills_to_learn:
        return {
            "algorithm": "A* Search (Optimal)",
            "initial_score": 100.0,
            "final_score": 100.0,
            "path": [],
            "total_effort_days": 0,
            "total_skills_to_learn": 0,
            "nodes_explored": 0,
            "iterations": 0,
            "trace": [{"step": 0, "action": "ALREADY AT GOAL",
                       "description": "Candidate already has all required skills!"}],
            "success": True
        }

    start_skills = frozenset(current_skills)
    start_h = heuristic(start_skills, all_needed)

    # Priority queue with A* nodes
    open_set = []
    start_node = AStarNode(f_score=start_h, g_score=0, skills=start_skills, path=())
    heapq.heappush(open_set, start_node)

    visited = set()
    iterations = 0
    nodes_explored = 0
    trace = []

    trace.append({
        "step": 0,
        "action": "INITIALIZE A*",
        "g_score": 0,
        "h_score": start_h,
        "f_score": start_h,
        "skills_to_learn": len(skills_to_learn),
        "description": f"A* initialized | g(start)=0, h(start)={start_h}, f(start)={start_h} | "
                       f"Need to learn {len(skills_to_learn)} skills"
    })

    while open_set and iterations < max_iterations:
        iterations += 1
        current = heapq.heappop(open_set)

        # ── Goal Check ──
        if all_needed.issubset(current.skills):
            # Reconstruct the solution path
            path = []
            total_effort = 0
            running_skills = set(start_skills)

            for idx, skill_name in enumerate(current.path, 1):
                running_skills.add(skill_name)
                score = calculate_match_score(running_skills, target_skills)
                info = SKILL_DATABASE.get(skill_name, {})
                effort = info.get("effort_days", 7)
                total_effort += effort

                path.append({
                    "step": idx,
                    "skill": skill_name,
                    "category": info.get("category", "Unknown"),
                    "effort_days": effort,
                    "score_after": round(score, 1),
                    "is_prerequisite": skill_name not in target_skills,
                    "description": info.get("description", skill_name)
                })

            trace.append({
                "step": iterations,
                "action": "✅ OPTIMAL GOAL REACHED",
                "nodes_explored": nodes_explored,
                "total_effort": total_effort,
                "path_length": len(current.path),
                "description": f"Optimal path found! {nodes_explored} nodes explored in "
                               f"{iterations} iterations | Total effort: {total_effort} days"
            })

            return {
                "algorithm": "A* Search (Optimal)",
                "initial_score": round(initial_score, 1),
                "final_score": 100.0,
                "path": path,
                "total_effort_days": total_effort,
                "total_skills_to_learn": len(path),
                "nodes_explored": nodes_explored,
                "iterations": iterations,
                "trace": trace,
                "success": True
            }

        # Skip already-visited states
        if current.skills in visited:
            continue
        visited.add(current.skills)
        nodes_explored += 1

        # ── Expand: try learning each available skill ──
        for skill in all_needed:
            if skill in current.skills:
                continue
            if skill not in SKILL_DATABASE:
                continue

            prereqs = set(SKILL_DATABASE[skill]["prerequisites"])
            if not prereqs.issubset(current.skills):
                continue

            new_skills = current.skills | frozenset([skill])
            if new_skills in visited:
                continue

            effort = SKILL_DATABASE[skill].get("effort_days", 7)
            new_g = current.g_score + effort
            new_h = heuristic(new_skills, all_needed)
            new_f = new_g + new_h

            new_path = current.path + (skill,)
            new_node = AStarNode(f_score=new_f, g_score=new_g,
                                 skills=new_skills, path=new_path)
            heapq.heappush(open_set, new_node)

        # Log progress periodically
        if nodes_explored <= 5 or nodes_explored % 100 == 0:
            trace.append({
                "step": iterations,
                "action": "EXPLORING",
                "nodes_explored": nodes_explored,
                "queue_size": len(open_set),
                "current_g": current.g_score,
                "current_f": current.f_score,
                "description": f"Explored {nodes_explored} nodes | Queue: {len(open_set)} | "
                               f"Current f={current.f_score}"
            })

    # Search exhausted or hit max iterations
    trace.append({
        "step": iterations,
        "action": "SEARCH EXHAUSTED",
        "nodes_explored": nodes_explored,
        "description": f"A* completed after {iterations} iterations, {nodes_explored} nodes explored. "
                       f"No complete path found within iteration limit."
    })

    return {
        "algorithm": "A* Search (Optimal)",
        "initial_score": round(initial_score, 1),
        "final_score": round(calculate_match_score(current_skills, target_skills), 1),
        "path": [],
        "total_effort_days": 0,
        "total_skills_to_learn": 0,
        "nodes_explored": nodes_explored,
        "iterations": iterations,
        "trace": trace,
        "success": False
    }


# ─── Compare Both Algorithms ─────────────────────────────────────────────────

def compare_algorithms(current_skills: Set[str], target_skills: Set[str]) -> dict:
    """
    Run both Hill Climbing and A* on the same problem and compare results.
    Demonstrates the trade-off between greedy speed and optimal completeness.
    """
    hill_result = hill_climbing_optimizer(current_skills, target_skills)
    astar_result = a_star_optimizer(current_skills, target_skills)

    hill_effort = hill_result["total_effort_days"]
    astar_effort = astar_result["total_effort_days"]

    if astar_effort > 0:
        effort_difference = hill_effort - astar_effort
        efficiency = round((effort_difference / astar_effort) * 100, 1)
    else:
        effort_difference = 0
        efficiency = 0

    return {
        "hill_climbing": hill_result,
        "a_star": astar_result,
        "comparison": {
            "hill_climbing_effort": hill_effort,
            "a_star_effort": astar_effort,
            "effort_saved_by_astar": effort_difference,
            "astar_efficiency_gain_percent": efficiency,
            "hill_climbing_steps": hill_result["total_skills_to_learn"],
            "a_star_steps": astar_result["total_skills_to_learn"],
            "hill_climbing_success": hill_result["success"],
            "a_star_success": astar_result["success"],
            "summary": (
                f"A* found a path requiring {astar_effort} days total effort "
                f"({astar_result['total_skills_to_learn']} skills). "
                f"Hill Climbing found a path requiring {hill_effort} days "
                f"({hill_result['total_skills_to_learn']} skills). "
                f"A* saves {effort_difference} days ({efficiency}% more efficient)."
            )
        }
    }
