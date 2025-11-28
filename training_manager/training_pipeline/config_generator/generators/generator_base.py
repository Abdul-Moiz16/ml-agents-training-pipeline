import itertools
import hashlib
from pathlib import Path
from random import randrange, randint, uniform

N_SAMPLES = 5

class ConfigGeneratorBase:

    def __init__(self, template_path, param_dict, output_dir, algorithm_name):
        self.batch_size = None
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


    def gen_params_dict(self):
        params = {}
        for key, values in self.param_dict.items():
            rand_value = self.randomize_value(key, values)
            params[key] = rand_value

            # Check if key = batch_size and if so save it for later calculation of buffer_size
            if key == "batch_size":
                self.batch_size = rand_value

        return params

    def randomize_value(self, key, values):
        # Key = buffer_size (needs to be = k x batch_size)
        if key == "buffer_size":
            return self.batch_size * randint(800, 14000) # TODO find optimal range for the multiple

        # Key has 1 value: return it
        elif len(values) == 1:
            return values[0]

        # Key has list of strings: randomly pick a string
        elif isinstance(values[0], str):
            rand_ind = randrange(len(values))
            return values[rand_ind]

        # Key has 2 integer value: randomize int value in between the 2 numbers
        elif len(values) == 2:
            # Values are integers
            if isinstance(values[0], int):
                return randint(values[0], values[1])

            # Values are floats
            if isinstance(values[0], float):
                return uniform(values[0], values[1])

        else:
            raise ValueError(f"Values for {key} must be a numeric range or a list of strings")



    def fill_template(self, params):
        txt = self.template_text
        for k, v in params.items():
            txt = txt.replace(f"{{{{{k}}}}}", str(v))
        return txt


    def generate(self):
        print(f"\n~ generating configs for {self.algorithm_name}")

        for _ in range(N_SAMPLES):
            params = self.gen_params_dict()

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
