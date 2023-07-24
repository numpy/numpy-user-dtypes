class Singleton:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


class NAType(Singleton):
    def __repr__(self):
        return "stringdtype.NA"


NA = NAType()
