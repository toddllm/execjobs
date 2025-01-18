import json
target_ciks = ['1652044', '320193', '789019', '14272', '1730168', '12927', '1521332', '1800', '1755672', '1467373']
with open('cik_to_company.json', 'r') as f:
    data = json.load(f)
    
# Get just the mappings we need
needed_mappings = {
    cik: data['cik_to_name'][cik.zfill(10)]
    for cik in target_ciks
}

print("\nMappings for our target companies:")
print(json.dumps(needed_mappings, indent=2))
