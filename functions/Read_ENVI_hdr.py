import re
class EnviHDR:
    def __init__(self, path):
        hdr_file = os.path.splitext(path)[0] + ".hdr"
        try:
            with open(hdr_file) as f:
                self._raw = f.readlines()
        except:
            ap.addErrorMessage("Cannot read header file at " +
                                path)
            pass
        self.tags = dict()
        
        iterator = iter(self._raw)
        values = None
        for iteration in iterator:
            line = iteration.replace("\n", "")
            if "=" in line:
                [tag, value] = [x.strip() for x in line.split("=", 1)]
                tag = re.sub("\s+", "_", tag)
                try:
                    value = int(value)
                    self.tags[tag] = value
                    continue
                except:
                    try:
                        value = float(value)
                        self.tags[tag] = value
                        continue
                    except:
                        pass
            else:
                continue
            if "{" in value:
                value = value.replace("{", "")
                if "}" in value:
                    value = value.replace("}", "")
                    values = [x.strip() for x in value.split(",")]
                else:
                    values = []
                    while True:
                        v = value.replace(",\n", "").replace("}", "") \
                            .strip()
                        if v != "":
                            for w in v.split(","):
                                try:
                                    w = int(w)
                                    values.append(w)
                                except:
                                    try:
                                        w = float(w)
                                        values.append(w)
                                    except:
                                        values.append(w.strip())
                        if "}" in value:
                            break
                        else:
                            try:
                                iteration = next(iterator)
                                value = iteration
                            except:
                                break
            self.tags[tag] = values
