from rule_check import parse_reply

REPLIES = {
 "1 good rule": '{"rule_id":"r_0001","object_class":"glass_jar","grip_force_min_n":14,"carry_accel_max_ms2":2.5,"rationale":"slip before drop","source_report_ids":["fr_0001"]}',
 "2 no safe rule": '{"no_safe_rule": true, "reason": "evidence too thin"}',
 "3 chatty + code fence": 'Sure! Here is the rule:\n```json\n{"rule_id":"r_0002"}\n```',
 "4 force too high": '{"rule_id":"r_0003","object_class":"glass_jar","grip_force_min_n":35,"carry_accel_max_ms2":2.5,"rationale":"grip hard","source_report_ids":["fr_0001"]}',
 "5 made-up class": '{"rule_id":"r_0004","object_class":"bottle","grip_force_min_n":10,"carry_accel_max_ms2":2.5,"rationale":"x","source_report_ids":["fr_0001"]}',
 "6 extra field": '{"rule_id":"r_0005","object_class":"steel","grip_force_min_n":10,"carry_accel_max_ms2":2.5,"rationale":"x","source_report_ids":["fr_0001"],"confidence":0.99}',
}

for name, text in REPLIES.items():
    obj, err = parse_reply(text)
    print(f"{name:24} -> {'ACCEPT ' + type(obj).__name__ if obj else 'REJECT: ' + err}")
