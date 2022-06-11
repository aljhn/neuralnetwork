def parse_config(config_file_path):
    config = {}

    f = open(config_file_path)
    lines = f.readlines()
    layer_amount = 0 # Number of layers found so far

    for line in lines:
        line = line.lower() # To lowercase
        line = line[:line.find("#")] # Remove comments
        line = "".join(line.split()) # Remove all whitespace

        if len(line) == 0: # Nothing here
            continue
        
        line = line.split(":")

        try:
            line[1] = float(line[1])
        except ValueError:
            pass

        if line[1] == "":
            layer_amount += 1
            config["layer" + str(layer_amount) + "_type"] = line[0]
        elif line[0][0] == "_":
            config["layer" + str(layer_amount) + line[0]] = line[1]
        else:
            config[line[0]] = line[1]

    f.close()

    config["layer_amount"] = layer_amount

    return config
