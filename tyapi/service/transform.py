from __future__ import annotations

import re
import textwrap
from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Optional, Set


class CycleError(RuntimeError):
    pass


def _topo_sort(nodes: Dict[str, Dict[str, Any]]) -> List[str]:
    indegree = {name: 0 for name in nodes}
    edges = defaultdict(list)
    for name, payload in nodes.items():
        for dep in payload.get("deps", []):
            if dep not in nodes:
                raise KeyError(f"Unknown dependency '{dep}' for node '{name}'")
            edges[dep].append(name)
            indegree[name] += 1
    queue = deque([name for name, deg in indegree.items() if deg == 0])
    order: List[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for child in edges[current]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(nodes):
        raise CycleError("Cycle detected in plan")
    return order


def _children_map(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    mapping = defaultdict(list)
    for name, payload in nodes.items():
        for dep in payload.get("deps", []):
            mapping[dep].append(name)
    return mapping


def _levelize(nodes: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    order = _topo_sort(nodes)
    levels: Dict[str, int] = {}
    for name in order:
        deps = nodes[name].get("deps", [])
        if not deps:
            levels[name] = 0
        else:
            levels[name] = 1 + max(levels[dep] for dep in deps)
    return levels


def _longest_path(nodes: Dict[str, Dict[str, Any]], sink: Optional[str] = None) -> List[str]:
    order = _topo_sort(nodes)
    children = _children_map(nodes)
    distance = {name: 0 for name in nodes}
    previous = {name: None for name in nodes}
    for name in order:
        for child in children[name]:
            if distance[name] + 1 > distance[child]:
                distance[child] = distance[name] + 1
                previous[child] = name
    target = sink if sink and sink in nodes else max(distance, key=distance.get)
    path: List[str] = []
    current: Optional[str] = target
    while current is not None:
        path.append(current)
        current = previous[current]
    return list(reversed(path))


_URL_RE = re.compile(r"(?i)\b((?:https?://|www\.)[^\s<>\"'\)\]]+)")


def extract_links_from_text(text: Optional[str]) -> List[str]:
    if not text:
        return []
    urls = _URL_RE.findall(text)
    normalized = []
    for url in urls:
        if url.lower().startswith("www."):
            normalized.append(f"http://{url}")
        else:
            normalized.append(url)
    seen: Set[str] = set()
    deduped: List[str] = []
    for url in normalized:
        if url not in seen:
            seen.add(url)
            deduped.append(url)
    return deduped


def merge_links(spec_links: Any, prompt_links: Iterable[str]) -> List[str]:
    merged: List[str] = []
    if isinstance(spec_links, list):
        for item in spec_links:
            if isinstance(item, str):
                merged.append(item)
    merged.extend(prompt_links)
    seen: Set[str] = set()
    result: List[str] = []
    for url in merged:
        if url not in seen:
            seen.add(url)
            result.append(url)
    return result


def _graph_analysis(nodes: Dict[str, Dict[str, Any]], prefer_sink: str = "executive_summary") -> Dict[str, Any]:
    topo_order = _topo_sort(nodes)
    children = _children_map(nodes)
    levels = _levelize(nodes)
    sources = [name for name, payload in nodes.items() if not payload.get("deps")]
    sinks = [name for name in nodes if not children[name]]

    in_degree = {name: len(payload.get("deps", [])) for name, payload in nodes.items()}
    out_degree = {name: len(children[name]) for name in nodes}

    wave_counts = defaultdict(int)
    for name, level in levels.items():
        wave_counts[level] += 1
    max_parallelism = max(wave_counts.values()) if wave_counts else 1
    max_parallel_waves = [level for level, count in wave_counts.items() if count == max_parallelism]

    critical_path = _longest_path(nodes, sink=prefer_sink)

    waves: List[List[str]] = []
    if wave_counts:
        for level in range(max(levels.values()) + 1):
            group = [name for name, lvl in levels.items() if lvl == level]
            group.sort()
            waves.append(group)

    edge_count = sum(in_degree.values())

    top_joins = sorted(in_degree.items(), key=lambda item: item[1], reverse=True)[:3]
    top_bottlenecks = sorted(out_degree.items(), key=lambda item: item[1], reverse=True)[:3]

    return {
        "topo_order": topo_order,
        "levels": levels,
        "waves": waves,
        "sources": sources,
        "sinks": sinks,
        "in_degree": in_degree,
        "out_degree": out_degree,
        "edge_count": edge_count,
        "critical_path": critical_path,
        "critical_path_length": len(critical_path),
        "max_parallelism": max_parallelism,
        "max_parallel_waves": max_parallel_waves,
        "top_joins": top_joins,
        "top_bottlenecks": top_bottlenecks,
    }


class PlanTransformer:
    """Produce enriched Tygent plans from ingested framework graphs."""

    def __init__(self, redundancy_mode: str = "inline") -> None:
        self.redundancy_mode = redundancy_mode

    def transform(self, spec: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        analysis = _graph_analysis(spec)
        levels = analysis["levels"]
        critical_nodes: Set[str] = set(analysis["critical_path"])

        steps: List[Dict[str, Any]] = []
        prefetch_by_step: Dict[str, List[str]] = {}

        for name, node in spec.items():
            prompt = node.get("prompt", "")
            deps = list(node.get("deps", []))
            links = merge_links(node.get("links", []), extract_links_from_text(prompt))
            level = levels.get(name, 0)
            metadata = dict(node.get("metadata", {}))
            kind = node.get("kind", "llm")

            prefetch_by_step[name] = links

            needs_redundancy = metadata.get("redundancy_hint")
            if needs_redundancy is None:
                needs_redundancy = bool(deps) or name.endswith("summary")
            redundancy = None
            if self.redundancy_mode != "off" and needs_redundancy:
                redundancy = {
                    "name": f"{name}_redundancy",
                    "depends_on": [name],
                    "prompt_template": textwrap.dedent(
                        """
                        You are a validation agent. Review the output of '{name}' alongside its dependencies {inputs}
                        and report any logical gaps or unsupported claims. Reply with 'OK' if the content passes scrutiny.
                        """
                    ).strip(),
                }

            tags = list(metadata.get("tags", []))
            if name in critical_nodes and "critical" not in tags:
                tags.append("critical")
            metadata["tags"] = tags
            metadata.setdefault("level", level)
            metadata.setdefault("links", links)
            metadata.setdefault("prompt", prompt)
            metadata.setdefault("kind", kind)
            metadata.setdefault("is_critical", name in critical_nodes)

            steps.append(
                {
                    "name": name,
                    "kind": kind,
                    "dependencies": deps,
                    "prompt": prompt,
                    "links": links,
                    "level": level,
                    "metadata": metadata,
                    "is_critical": name in critical_nodes,
                    "tags": tags,
                    "redundancy": redundancy,
                }
            )

        if self.redundancy_mode == "steps":
            materialized: List[Dict[str, Any]] = []
            for step in steps:
                materialized.append(step)
                redundancy = step.get("redundancy")
                if redundancy:
                    materialized.append(
                        {
                            "name": redundancy["name"],
                            "kind": "llm",
                            "dependencies": redundancy.get("depends_on", []),
                            "prompt": redundancy.get("prompt_template", ""),
                            "links": [],
                            "level": None,
                            "metadata": {"generated": True, "source": step["name"]},
                            "is_critical": False,
                            "tags": ["redundancy"],
                            "redundancy": None,
                        }
                    )
            steps = materialized

        unique_links: List[str] = []
        seen_links: Set[str] = set()
        for link_list in prefetch_by_step.values():
            for url in link_list:
                if url not in seen_links:
                    seen_links.add(url)
                    unique_links.append(url)

        return {
            "name": "tygent_ingested_plan",
            "steps": steps,
            "prefetch": {
                "links": unique_links,
                "by_step": prefetch_by_step,
            },
            "meta": {
                "redundancy_mode": self.redundancy_mode,
                **analysis,
            },
        }
