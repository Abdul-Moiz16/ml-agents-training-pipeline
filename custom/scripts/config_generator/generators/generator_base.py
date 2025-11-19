import itertools
import hashlib
from pathlib import Path

class ConfigGeneratorBase:

    def __init__(self, template_path, param_dict, output_dir, algorithm_name):
        self.template_path = Path(template_path)
        self.template_text = self.template_path.read_text()
        self.param_dict = param_dict
        self.output_dir = Path(output_dir)
        self.algorithm_name = algorithm_name

        self.output_dir.mkdir(parents=True, exist_ok=True)


    def make_param_string(self, params):
        return ";".join(f"{k}={params[k]}" for k in sorted(params.keys()))

    def make_hash(self, s):
        return hashlib.sha1(s.encode()).hexdigest()[:10]

    def generate_combinations(self):
        keys = list(self.param_dict.keys())
        values = list(self.param_dict.values())

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def fill_template(self, params):
        txt = self.template_text
        for k, v in params.items():
            txt = txt.replace(f"{{{{{k}}}}}", str(v))
        return txt


    def generate(self):
        print(f"\n~ generating configs for {self.algorithm_name}")

        for params in self.generate_combinations():

            p_string = self.make_param_string(params)
            p_hash = self.make_hash(p_string)

            filename = f"{self.algorithm_name}_{p_hash}.yaml"
            outfile = self.output_dir / filename

            # skip duplicates
            if outfile.exists():
                print("skip (exists):", filename)
                continue

            # fill template + write
            content = self.fill_template(params)
            outfile.write_text(content)


            print("created:", filename)
