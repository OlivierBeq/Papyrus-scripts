"""initial

Revision ID: 436aabc6e771
Revises: 
Create Date: 2022-05-26 10:07:13.713675

"""
from alembic import op
import sqlalchemy as sa
import razi


# revision identifiers, used by Alembic.
revision = '436aabc6e771'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('activity_type',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('type', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('classification',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('classification', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('molecule',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('smiles', sa.Text(), nullable=True),
    sa.Column('mol', razi.rdkit_postgresql.types.Mol(), nullable=True),
    sa.Column('fp', razi.rdkit_postgresql.types.Bfp(), nullable=True),
    sa.Column('connectivity', sa.Text(), nullable=True),
    sa.Column('inchi_key', sa.Text(), nullable=True),
    sa.Column('inchi', sa.Text(), nullable=True),
    sa.Column('inchi_auxinfo', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('organism',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('organism', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('quality',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('quality', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('source',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('source', sa.Text(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('source')
    )
    op.create_table('cid',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('cid', sa.Text(), nullable=True),
    sa.Column('source', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['source'], ['source.source'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_table('protein',
    sa.Column('target_id', sa.Text(), nullable=False),
    sa.Column('HGNC_symbol', sa.Text(), nullable=True),
    sa.Column('uniprot_id', sa.Text(), nullable=True),
    sa.Column('reviewed', sa.Boolean(), nullable=True),
    sa.Column('organism', sa.Integer(), nullable=True),
    sa.Column('classification', sa.Integer(), nullable=True),
    sa.Column('length', sa.Integer(), nullable=True),
    sa.Column('sequence', sa.Text(), nullable=True),
    sa.ForeignKeyConstraint(['classification'], ['classification.id'], ),
    sa.ForeignKeyConstraint(['organism'], ['organism.id'], ),
    sa.PrimaryKeyConstraint('target_id')
    )
    op.create_table('activity',
    sa.Column('activity_id', sa.Integer(), nullable=False),
    sa.Column('quality', sa.Integer(), nullable=True),
    sa.Column('target_id', sa.Text(), nullable=True),
    sa.Column('molecule_id', sa.Integer(), nullable=True),
    sa.Column('aid', sa.Text(), nullable=True),
    sa.Column('doc_id', sa.Text(), nullable=True),
    sa.Column('year', sa.Integer(), nullable=True),
    sa.Column('all_doc_ids', sa.Text(), nullable=True),
    sa.Column('all_years', sa.Text(), nullable=True),
    sa.Column('type', sa.Integer(), nullable=True),
    sa.Column('relation', sa.Text(), nullable=True),
    sa.Column('pchembl_value', sa.Float(), nullable=True),
    sa.Column('pchembl_value_mean', sa.Float(), nullable=True),
    sa.Column('pchembl_value_stdev', sa.Float(), nullable=True),
    sa.Column('pchembl_value_SEM', sa.Float(), nullable=True),
    sa.Column('pchembl_value_n', sa.Float(), nullable=True),
    sa.Column('pchembl_value_median', sa.Float(), nullable=True),
    sa.Column('pchembl_value_mad', sa.Float(), nullable=True),
    sa.ForeignKeyConstraint(['molecule_id'], ['molecule.id'], ),
    sa.ForeignKeyConstraint(['quality'], ['quality.id'], ),
    sa.ForeignKeyConstraint(['target_id'], ['protein.target_id'], ),
    sa.ForeignKeyConstraint(['type'], ['activity_type.id'], ),
    sa.PrimaryKeyConstraint('activity_id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('activity')
    op.drop_table('protein')
    op.drop_table('cid')
    op.drop_table('source')
    op.drop_table('quality')
    op.drop_table('organism')
    op.drop_table('molecule')
    op.drop_table('classification')
    op.drop_table('activity_type')
    # ### end Alembic commands ###
