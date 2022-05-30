from ast import For
# from xmlrpc.client import Boolean
from sqlalchemy import Column, Integer, Text, Float, ForeignKey, Boolean, Table
from sqlalchemy.orm import relationship
from razi.rdkit_postgresql.types import Mol, Bfp

from .session import Base


# association table between protein and classification (Many to Many relationship)
ProteinClassification = Table('ProteinClassification',
Base.metadata,
Column('id', Integer, primary_key=True),
Column('protein_id', Text, ForeignKey('protein.target_id')),
Column('classification_id', Integer, ForeignKey('classification.id'))
)


class Protein(Base):
    __tablename__ = "protein"

    target_id = Column(Text, primary_key=True)
    HGNC_symbol = Column(Text, nullable=True)
    uniprot_id = Column(Text)
    reviewed = Column(Boolean)
    organism = Column(Integer, ForeignKey('organism.id'))
    classifications = relationship('Classification', secondary=ProteinClassification, backref='protein')
    length = Column(Integer)
    sequence = Column(Text)


class Organism(Base):
    __tablename__ = "organism"

    id = Column(Integer, primary_key=True)
    organism = Column(Text)


class Classification(Base):
    __tablename__ = "classification"

    id = Column(Integer, primary_key=True)
    classification = Column(Text)
    proteins = relationship('Protein', secondary=ProteinClassification, backref='classification')


class Molecule(Base):
    __tablename__ = "molecule"

    id = Column(Integer, primary_key=True) # assign a new integer value for every molecule since it may have multiple CID records
    cids = relationship('CID', back_populates='molecule')
    smiles = Column(Text) #isosmiles?
    mol = Column(Mol)
    fp = Column(Bfp)
    connectivity = Column(Text)
    inchi_key = Column(Text)
    inchi = Column(Text)
    inchi_auxinfo = Column(Text)


class Activity(Base):
    __tablename__ = 'activity'
    # CHANGE TO id AND activity_id before deploying
    activity_id = Column(Integer, primary_key=True)
    papyrus_activity_id = Column(Text)
    quality = Column(Integer, ForeignKey('quality.id'))
    # source = Column(Integer, ForeignKey('source.id')) # this can be recovered from CID table
    target_id = Column(Text, ForeignKey('protein.target_id'))
    accession = Column(Text)
    protein_type = Column(Text)
    molecule_id = Column(Integer, ForeignKey('molecule.id')) # use first value here
    # alt_cid_source_pairs = relationship('cid') # get this from molecule info
    aid = Column(Text)
    doc_id = Column(Text)
    year = Column(Integer)
    # all_doc_ids = Column(Text)
    # all_years = Column(Text)
    type = Column(Integer, ForeignKey('activity_type.id'))
    # activity_class?
    relation = Column(Text)
    pchembl_value = Column(Float)
    pchembl_value_mean = Column(Float)
    pchembl_value_stdev = Column(Float)
    pchembl_value_SEM = Column(Float)
    pchembl_value_n = Column(Float)
    pchembl_value_median = Column(Float)
    pchembl_value_mad = Column(Float)


class ActivityType(Base):
    __tablename__ = "activity_type"

    id = Column(Integer, primary_key=True)
    type = Column(Text)


class Source(Base):
    __tablename__="source"

    id = Column(Integer, primary_key=True)
    source = Column(Text, unique=True)


class Quality(Base):
    __tablename__="quality"

    id = Column(Integer, primary_key=True)
    quality = Column(Text)


class CID(Base):
    __tablename__="cid"

    id = Column(Integer, primary_key=True)
    cid = Column(Text)
    source = Column(Text, ForeignKey('source.source'))
    molecule = relationship('Molecule', back_populates='cids')
    molecule_id = Column(Integer, ForeignKey('molecule.id'))