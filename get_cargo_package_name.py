import pytoml as toml
with open('Cargo.toml', 'rb') as fin:
    obj = toml.load(fin)

print(obj["package"]["name"])
