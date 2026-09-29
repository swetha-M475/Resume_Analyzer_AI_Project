"""
Job Route Optimizer Module
Solves the Traveling Salesman Problem (TSP) for optimal job application route planning.

AI Concepts Demonstrated:
    - TSP (Traveling Salesman Problem): NP-hard combinatorial optimization
    - Brute Force: Exact solution by checking all permutations — O(n!)
    - Nearest Neighbor Heuristic: Greedy approximation — O(n²)
    - Dynamic Programming (Held-Karp): Optimal for moderate inputs — O(n² × 2^n)
"""

import itertools
import math
import time
from typing import List, Dict, Tuple
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

# Initialize geocoder (requires a custom user_agent per Nominatim usage policy)
geolocator = Nominatim(user_agent="resume_analyzer_job_route")

# Cache to store dynamic coordinates and avoid redundant API calls
_GEO_CACHE: Dict[str, Tuple[float, float]] = {}


# ─── Predefined City Coordinates (Lat, Lon) ──────────────────────────────────
# Covers major Indian tech hubs and global cities

CITY_COORDINATES = {
    # Chennai Locations
    "tidel park, chennai": (12.9897, 80.2483),
    "omr, chennai": (12.9056, 80.2263),
    "guindy, chennai": (13.0067, 80.2206),
    "siruseri, chennai": (12.8288, 80.2185),
    "tambaram, chennai": (12.9249, 80.1000),

    # Bangalore Locations
    "electronic city, bangalore": (12.8452, 77.6601),
    "whitefield, bangalore": (12.9698, 77.7499),
    "koramangala, bangalore": (12.9279, 77.6271),
    "indiranagar, bangalore": (12.9718, 77.6411),
    "manyata tech park, bangalore": (13.0450, 77.6205),

    # Coimbatore Locations
    "saravanampatti, coimbatore": (11.0805, 76.9946),
    "peelamedu, coimbatore": (11.0261, 77.0121),
    "rs puram, coimbatore": (11.0094, 76.9482),
    "tidels park, coimbatore": (11.0264, 77.0264),
    "gandhipuram, coimbatore": (11.0180, 76.9659),

    # Hyderabad Locations
    "hitec city, hyderabad": (17.4474, 78.3762),
    "gachibowli, hyderabad": (17.4401, 78.3489),
    "madhapur, hyderabad": (17.4483, 78.3915),
    "banjara hills, hyderabad": (17.4156, 78.4347),
    "kukatpally, hyderabad": (17.4842, 78.4069),
}


def haversine_distance(coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
    """
    Calculate the great-circle distance between two points on Earth.
    Uses the Haversine formula.

    Args:
        coord1: (latitude, longitude) of point 1
        coord2: (latitude, longitude) of point 2

    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth's radius in kilometers

    lat1, lon1 = math.radians(coord1[0]), math.radians(coord1[1])
    lat2, lon2 = math.radians(coord2[0]), math.radians(coord2[1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.asin(math.sqrt(a))

    return round(R * c, 1)


def get_city_coordinates(city_name: str) -> Tuple[float, float]:
    """
    Retrieve coordinates for a given city or address.
    1. Checks exact match in local CITY_COORDINATES.
    2. Checks dynamic cache.
    3. Pings OpenStreetMap via geopy if not found locally.
    
    Returns (latitude, longitude) or None if not found.
    """
    if not city_name:
        return None

    normalized = city_name.strip().lower()

    if normalized in CITY_COORDINATES:
        return CITY_COORDINATES[normalized]

    if normalized in _GEO_CACHE:
        return _GEO_CACHE[normalized]

    try:
        # Nominatim requests shouldn't exceed 1 request per second
        time.sleep(1) 
        location = geolocator.geocode(city_name)
        
        if location:
            coords = (location.latitude, location.longitude)
            _GEO_CACHE[normalized] = coords
            return coords
            
    except (GeocoderTimedOut, GeocoderServiceError) as e:
        print(f"Geopy error for {city_name}: {e}")
        pass

    return None


def build_distance_matrix(jobs: List[Dict]) -> Tuple[List[List[float]], List[Tuple[float, float]]]:
    """
    Build a distance matrix between all job locations using Haversine formula.

    Args:
        jobs: List of job dictionaries with 'city' field

    Returns:
        Tuple of (distance_matrix, coordinates_list)
    """
    n = len(jobs)
    matrix = [[0.0] * n for _ in range(n)]
    coords = []

    for job in jobs:
        city = job.get("city", "").strip()
        coord = get_city_coordinates(city)
        if coord:
            coords.append(coord)
        elif "lat" in job and "lon" in job:
            coords.append((float(job["lat"]), float(job["lon"])))
        else:
            coords.append((0, 0))

    for i in range(n):
        for j in range(i + 1, n):
            dist = haversine_distance(coords[i], coords[j])
            matrix[i][j] = dist
            matrix[j][i] = dist

    return matrix, coords


# ─── Brute Force TSP (Exact Solution) ────────────────────────────────────────

def tsp_brute_force(distance_matrix: List[List[float]], job_names: List[str]) -> dict:
    """
    Solve TSP using Brute Force — checks ALL permutations to find the optimal tour.

    Time Complexity: O(n!) — only feasible for n ≤ 10
    Space Complexity: O(n)

    This guarantees the OPTIMAL (shortest) route but is extremely slow for large inputs.

    Args:
        distance_matrix: 2D distance matrix
        job_names: List of job/city names

    Returns:
        Dictionary with optimal route, distance, and execution trace
    """
    n = len(distance_matrix)

    if n <= 1:
        return {
            "algorithm": "Brute Force (Exact/Optimal)",
            "route": list(range(n)),
            "route_names": job_names[:],
            "total_distance_km": 0,
            "trace": [{"step": 0, "description": "Only one location — no route needed."}],
            "permutations_checked": 1
        }

    cities = list(range(n))
    best_route = None
    best_distance = float('inf')
    trace = []
    perms_checked = 0

    # Fix first city (city 0) to reduce permutations by factor of n
    # Since the tour is a cycle, starting point doesn't matter
    for perm in itertools.permutations(cities[1:]):
        route = [cities[0]] + list(perm)
        total_dist = 0

        # Calculate tour length
        for i in range(len(route) - 1):
            total_dist += distance_matrix[route[i]][route[i + 1]]
        # Return to start (complete the cycle)
        total_dist += distance_matrix[route[-1]][route[0]]
        total_dist = round(total_dist, 1)

        perms_checked += 1

        if total_dist < best_distance:
            best_distance = total_dist
            best_route = route[:]
            trace.append({
                "step": perms_checked,
                "route": [job_names[i] for i in route],
                "distance_km": total_dist,
                "description": f"New best route found! Distance: {total_dist} km "
                               f"(checked {perms_checked} permutations so far)"
            })

    route_names = [job_names[i] for i in best_route]

    # Build segment details
    segments = []
    for i in range(len(best_route)):
        from_idx = best_route[i]
        to_idx = best_route[(i + 1) % len(best_route)]
        dist = distance_matrix[from_idx][to_idx]
        segments.append({
            "from": job_names[from_idx],
            "to": job_names[to_idx],
            "distance_km": dist
        })

    return {
        "algorithm": "Brute Force (Exact/Optimal)",
        "route": best_route,
        "route_names": route_names,
        "total_distance_km": best_distance,
        "segments": segments,
        "permutations_checked": perms_checked,
        "total_permutations": math.factorial(n - 1),
        "trace": trace
    }


# ─── Nearest Neighbor Heuristic (Greedy TSP) ─────────────────────────────────

def tsp_nearest_neighbor(distance_matrix: List[List[float]], job_names: List[str],
                         start: int = 0) -> dict:
    """
    Solve TSP using the Nearest Neighbor heuristic.

    Algorithm:
        1. Start at the given city
        2. At each step, visit the CLOSEST unvisited city
        3. Repeat until all cities are visited
        4. Return to the starting city

    Time Complexity: O(n²) — much faster than brute force
    Space Complexity: O(n)

    This is a GREEDY approach — fast but may not find the optimal route.

    Args:
        distance_matrix: 2D distance matrix
        job_names: List of job/city names
        start: Starting city index

    Returns:
        Dictionary with route, distance, and execution trace
    """
    n = len(distance_matrix)

    if n <= 1:
        return {
            "algorithm": "Nearest Neighbor (Greedy Heuristic)",
            "route": list(range(n)),
            "route_names": job_names[:],
            "total_distance_km": 0,
            "trace": [{"step": 0, "description": "Only one location."}]
        }

    visited = [False] * n
    route = [start]
    visited[start] = True
    total_distance = 0
    trace = []

    trace.append({
        "step": 0,
        "action": f"START at {job_names[start]}",
        "total_distance_km": 0,
        "description": f"Begin tour at {job_names[start]}"
    })

    for step_num in range(1, n):
        current = route[-1]
        nearest = None
        nearest_dist = float('inf')

        # Find all unvisited candidates and their distances
        candidates = []
        for j in range(n):
            if not visited[j]:
                dist = distance_matrix[current][j]
                candidates.append({
                    "city": job_names[j],
                    "distance_km": dist
                })
                if dist < nearest_dist:
                    nearest_dist = dist
                    nearest = j

        route.append(nearest)
        visited[nearest] = True
        total_distance += nearest_dist

        # Sort candidates by distance for trace readability
        candidates.sort(key=lambda x: x["distance_km"])

        trace.append({
            "step": step_num,
            "action": f"VISIT → {job_names[nearest]}",
            "from": job_names[current],
            "to": job_names[nearest],
            "leg_distance_km": round(nearest_dist, 1),
            "total_distance_km": round(total_distance, 1),
            "candidates_considered": len(candidates),
            "nearest_candidate": candidates[0] if candidates else None,
            "description": f"From {job_names[current]} → {job_names[nearest]} "
                           f"({round(nearest_dist, 1)} km) | "
                           f"Total so far: {round(total_distance, 1)} km | "
                           f"Chose nearest from {len(candidates)} options"
        })

    # Return to start
    return_dist = distance_matrix[route[-1]][route[0]]
    total_distance += return_dist

    trace.append({
        "step": n,
        "action": f"RETURN → {job_names[start]}",
        "from": job_names[route[-1]],
        "to": job_names[start],
        "leg_distance_km": round(return_dist, 1),
        "total_distance_km": round(total_distance, 1),
        "description": f"Return to start: {job_names[route[-1]]} → {job_names[start]} "
                       f"({round(return_dist, 1)} km) | "
                       f"Total tour: {round(total_distance, 1)} km"
    })

    route_names = [job_names[i] for i in route]

    # Build segment details
    segments = []
    for i in range(len(route)):
        from_idx = route[i]
        to_idx = route[(i + 1) % len(route)]
        dist = distance_matrix[from_idx][to_idx]
        segments.append({
            "from": job_names[from_idx],
            "to": job_names[to_idx],
            "distance_km": dist
        })

    return {
        "algorithm": "Nearest Neighbor (Greedy Heuristic)",
        "route": route,
        "route_names": route_names,
        "total_distance_km": round(total_distance, 1),
        "segments": segments,
        "trace": trace
    }


# ─── Dynamic Programming TSP (Held-Karp) ─────────────────────────────────────

def tsp_dynamic_programming(distance_matrix: List[List[float]], job_names: List[str]) -> dict:
    """
    Solve TSP using Dynamic Programming (Held-Karp algorithm).

    Algorithm:
        - Uses bitmask to represent subsets of visited cities
        - dp[mask][i] = minimum distance to visit all cities in mask, ending at city i
        - Builds up from smaller subsets to the full set

    Time Complexity: O(n² × 2^n) — much better than O(n!) for moderate n
    Space Complexity: O(n × 2^n)

    Feasible for n ≤ 20 (vs n ≤ 10 for brute force).

    Args:
        distance_matrix: 2D distance matrix
        job_names: List of job/city names

    Returns:
        Dictionary with optimal route, distance, and execution trace
    """
    n = len(distance_matrix)

    if n <= 1:
        return {
            "algorithm": "Dynamic Programming (Held-Karp, Optimal)",
            "route": list(range(n)),
            "route_names": job_names[:],
            "total_distance_km": 0,
            "trace": [{"step": 0, "description": "Only one location."}]
        }

    INF = float('inf')
    # dp[mask][i] = minimum cost to visit all cities in 'mask', ending at city i
    # Starting city is fixed at 0
    full_mask = (1 << n) - 1
    dp = [[INF] * n for _ in range(1 << n)]
    parent = [[-1] * n for _ in range(1 << n)]

    # Base case: start at city 0
    dp[1 << 0][0] = 0
    states_computed = 0
    trace = []

    trace.append({
        "step": 0,
        "action": "INITIALIZE DP TABLE",
        "table_size": f"{1 << n} × {n}",
        "total_states": (1 << n) * n,
        "description": f"DP table size: {1 << n} subsets × {n} cities = "
                       f"{(1 << n) * n} states to evaluate"
    })

    # Fill DP table
    for mask in range(1 << n):
        for u in range(n):
            if dp[mask][u] == INF:
                continue
            if not (mask & (1 << u)):
                continue

            # Try extending to each unvisited city
            for v in range(n):
                if mask & (1 << v):
                    continue  # Already visited

                new_mask = mask | (1 << v)
                new_cost = dp[mask][u] + distance_matrix[u][v]

                if new_cost < dp[new_mask][v]:
                    dp[new_mask][v] = new_cost
                    parent[new_mask][v] = u
                    states_computed += 1

    # Find the best ending city (return to start)
    best_cost = INF
    best_last = -1
    for u in range(n):
        total = dp[full_mask][u] + distance_matrix[u][0]
        if total < best_cost:
            best_cost = total
            best_last = u

    # Reconstruct path
    path = []
    mask = full_mask
    current = best_last

    while current != -1:
        path.append(current)
        prev = parent[mask][current]
        mask = mask ^ (1 << current)
        current = prev

    path.reverse()
    route_names = [job_names[i] for i in path]

    # Build segment details
    segments = []
    for i in range(len(path)):
        from_idx = path[i]
        to_idx = path[(i + 1) % len(path)]
        dist = distance_matrix[from_idx][to_idx]
        segments.append({
            "from": job_names[from_idx],
            "to": job_names[to_idx],
            "distance_km": dist
        })

    trace.append({
        "step": 1,
        "action": "DP TABLE FILLED",
        "states_computed": states_computed,
        "description": f"Computed {states_computed} state transitions"
    })

    trace.append({
        "step": 2,
        "action": "✅ OPTIMAL TOUR FOUND",
        "total_distance_km": round(best_cost, 1),
        "route": route_names,
        "description": f"Optimal tour distance: {round(best_cost, 1)} km"
    })

    return {
        "algorithm": "Dynamic Programming (Held-Karp, Optimal)",
        "route": path,
        "route_names": route_names,
        "total_distance_km": round(best_cost, 1),
        "segments": segments,
        "states_computed": states_computed,
        "trace": trace
    }


# ─── Main Solver ─────────────────────────────────────────────────────────────

def solve_tsp(jobs: List[Dict]) -> dict:
    """
    Solve the TSP for a list of job locations.

    Strategy:
        - For n ≤ 8: Run Brute Force (exact) + Nearest Neighbor (heuristic)
        - For 8 < n ≤ 15: Run DP Held-Karp (exact) + Nearest Neighbor (heuristic)
        - For n > 15: Run Nearest Neighbor only (heuristic)

    This allows comparing exact vs heuristic solutions on small inputs.

    Args:
        jobs: List of job dicts with 'name', 'city', 'company' fields

    Returns:
        Dictionary with solutions and comparisons
    """
    if not jobs:
        return {"error": "No jobs provided"}

    n = len(jobs)
    job_names = []
    for j in jobs:
        company = j.get("company", "Company")
        city = j.get("city", "Unknown")
        job_names.append(f"{company} ({city})")

    distance_matrix, coordinates = build_distance_matrix(jobs)

    results = {
        "jobs": jobs,
        "job_names": job_names,
        "distance_matrix": distance_matrix,
        "coordinates": coordinates,
        "num_jobs": n
    }

    # Always run Nearest Neighbor
    results["nearest_neighbor"] = tsp_nearest_neighbor(distance_matrix, job_names)

    # Run exact algorithm based on size
    if n <= 8:
        results["exact"] = tsp_brute_force(distance_matrix, job_names)
        results["exact_method"] = "Brute Force"
    elif n <= 15:
        results["exact"] = tsp_dynamic_programming(distance_matrix, job_names)
        results["exact_method"] = "Dynamic Programming (Held-Karp)"
    else:
        results["exact"] = None
        results["exact_method"] = None

    # Comparison
    nn_dist = results["nearest_neighbor"]["total_distance_km"]

    if results["exact"]:
        exact_dist = results["exact"]["total_distance_km"]

        if exact_dist > 0:
            optimality_gap = round(((nn_dist - exact_dist) / exact_dist) * 100, 1)
        else:
            optimality_gap = 0

        results["comparison"] = {
            "nearest_neighbor_km": nn_dist,
            "exact_km": exact_dist,
            "optimality_gap_percent": optimality_gap,
            "exact_method": results["exact_method"],
            "summary": f"Nearest Neighbor: {nn_dist} km | "
                       f"{results['exact_method']}: {exact_dist} km | "
                       f"Gap: {optimality_gap}% longer than optimal"
        }
    else:
        results["comparison"] = {
            "nearest_neighbor_km": nn_dist,
            "exact_km": None,
            "note": f"Exact solution skipped — {n} cities would require "
                    f"{math.factorial(n - 1):,} permutations for brute force. "
                    f"Only the Nearest Neighbor heuristic was used."
        }

    return results
