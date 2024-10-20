
def convert_params(config):
    for key, value in config.items():
        try:
            config[key] = int(value)
        except ValueError:
            pass
        except TypeError:
            pass
    return config