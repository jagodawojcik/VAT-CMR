from enum import Enum

# Modality Selection
class QueryModality(Enum):
    VISUAL = 'visual'
    TACTILE = 'tactile'
    AUDIO = 'audio'

class DominatingModality(Enum):
    VISUAL = 'visual'
    TACTILE = 'tactile'
    AUDIO = 'audio'
    JOINT = 'joint_embedding'


