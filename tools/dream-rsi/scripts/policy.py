"""Pure scheduling policy. No filesystem, tools, imports, or hidden outcomes."""


def schedule(beta, config):
    return {
        "width": max(1, round(1 + beta * (config["max_width"] - 1))),
        "depth": max(1, round(1 + beta * (config["max_depth"] - 1))),
        "patience": 1 + round(2 * beta),
        "repairs": 1 + int(beta >= 0.75),
    }


def choose(observed, legal, config, beta):
    limits = schedule(beta, config)
    ranked = []
    branch_count = len(set(node["branch"] for node in observed))
    scores = [node["score"] for node in observed if node["valid"]]
    best = max(scores) if scores else 0.0
    for action in legal:
        if action["parent"] == "root":
            if branch_count < limits["width"]:
                ranked.append((2.0, action["id"], "explore"))
            continue
        history = [node for node in observed if node["branch"] == action["branch"]]
        latest = history[-1]
        if latest["depth"] >= limits["depth"]:
            continue
        valid = [node for node in history if node["valid"]]
        anchor = max([node["score"] for node in valid]) if valid else None
        if not latest["valid"]:
            last_success = max([i for i in range(len(history)) if history[i]["valid"]], default=-1)
            failures = len(history) - 1 - last_success
            if latest["repairable"] and failures <= limits["repairs"]:
                ranked.append((1.5 + (0.25 if anchor is not None else 0), action["id"], "repair"))
            continue
        best_index = max(range(len(valid)), key=lambda i: (valid[i]["score"], -i))
        stale = len(valid) - 1 - best_index
        if stale >= limits["patience"]:
            continue
        gain = len(valid) == 1 or valid[-1]["score"] > valid[-2]["score"]
        priority = (3.0 if gain else 1.0) + (0.5 if anchor == best else 0)
        ranked.append((priority, action["id"], "refine"))
    ranked = sorted(ranked, key=lambda item: (-item[0], item[1]))
    batch = []
    repairs = 0
    for item in ranked:
        if item[2] == "repair" and repairs:
            continue
        batch.append(item[1])
        repairs += int(item[2] == "repair")
        if len(batch) >= config["workers"]:
            break
    return batch
