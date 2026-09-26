import json

with open("sample_pairs.json", "r", encoding="utf-8") as f:
    pairs = json.load(f)

with open("docs/sample_pairs.txt", "w", encoding="utf-8") as out:
    for p in pairs:
        out.write(f"Pair {p['pair_num']}: S1 ({p['s1_id']}) <-> Match ({p['match_id']})\n")
        out.write(f"  S1 Name:    {p['s1_name']}\n")
        out.write(f"  Match Name: {p['match_name']}\n")
        out.write(f"  S1 Addr:    {p['s1_addr']}\n")
        out.write(f"  Match Addr: {p['match_addr']}\n")
        out.write(f"  S1 Country: {p['s1_country']} | Match Country: {p['match_country']}\n")
        out.write("-" * 60 + "\n")

print(f"Successfully generated docs/sample_pairs.txt with {len(pairs)} pairs")
