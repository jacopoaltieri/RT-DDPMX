import yaml

def load_yaml(path):
    with open(path) as file:
        try:
            cfg = yaml.safe_load(file)
            return cfg
        except yaml.YAMLError as exc:
            print(exc)
    
if __name__ == "__main__":
    pass