# Papyrus postgres database

This code is intended to take the target, activity and molecule data from the papyrus author's recommended dataset and put it into a postgres database for querying.

It uses sqlalchemy models to define tables, fields and data types. The intention is that an API or pythonic querying can therefore be implemented to interact with the data through either scripts, jupyter notebooks or web interfaces.

In addition, the rdkit cartridge has been utilised to include rdkit Mol and Fingerprint objects