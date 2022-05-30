"""association

Revision ID: 2d33c9b9a809
Revises: a56cddd55ba6
Create Date: 2022-05-26 11:12:38.050525

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '2d33c9b9a809'
down_revision = 'a56cddd55ba6'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ProteinClassification',
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('protein_id', sa.Text(), nullable=True),
    sa.Column('classification_id', sa.Integer(), nullable=True),
    sa.ForeignKeyConstraint(['classification_id'], ['classification.id'], ),
    sa.ForeignKeyConstraint(['protein_id'], ['protein.target_id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ProteinClassification')
    # ### end Alembic commands ###
