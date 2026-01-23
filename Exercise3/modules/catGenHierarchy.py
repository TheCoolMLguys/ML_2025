import src.io.jsonInput as custom_json

class CatGenHierarchy:

    def __init__(self, label, genSource):
        """
        genSource can be:
        - str  -> path to JSON file
        - dict -> dynamic flat hierarchy {value: '*'}
        """
        self._label = label
        self._entries = {}
        self._levels = 0

        if isinstance(genSource, dict):
            self._buildFromDict(genSource)
        else:
            self.readFromJSON(genSource)

    # -------------------------------------------------

    def _buildFromDict(self, gen_dict):
        """
        Build a flat hierarchy dynamically:
            value -> '*'
        """
        idx = 0

        # Root
        self._entries["*"] = {
            "level": 0,
            "name": "*",
            "gen": "*"
        }

        # Leaves
        for val in gen_dict:
            self._entries[val] = {
                "level": 1,
                "name": str(val),
                "gen": "*"
            }

        self._levels = 1

    # -------------------------------------------------

    def readFromJSON(self, json_file):
        json_struct = custom_json.readJSON(json_file)
        entries = json_struct.get('entries')
        root_levels = 0

        for idx in entries:
            json_entry = entries[idx]
            level = int(json_entry.get('level'))

            self._levels = max(self._levels, level)

            if level == 0:
                root_levels += 1

            self._entries[idx] = {
                'level': level,
                'name': json_entry.get('name'),
                'gen': json_entry.get('gen')
            }

        if root_levels != 1:
            raise Exception('JSON invalid. Level 0 must occur exactly once.')

    # -------------------------------------------------

    def getEntries(self):
        return self._entries

    def nrLevels(self):
        return self._levels + 1   # include level 0

    def getLevelEntry(self, key):
        return self._entries[key]['level']

    def getGeneralizationOf(self, key):
        return self._entries[key]['gen']

    def getNameEntry(self, key):
        return self._entries[key]['name']
