from enum import Enum

# Modality Selection
class QueryModality(Enum):
    VISUAL = 'visual'
    TACTILE = 'tactile'
    AUDIO = 'audio'

class DominatingModality(QueryModality):
    JOINT = 'joint_embedding'


