class MetadataScalar:
    def __init__(self, value, metadata):
        self.value = value
        self.metadata = metadata

    def __repr__(self):
        return f"{self.value} {self.metadata}"
