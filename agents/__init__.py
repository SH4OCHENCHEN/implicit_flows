from agents.c51 import C51Agent
from agents.codac import CODACAgent
from agents.fbrac import FBRACAgent
from agents.fql import FQLAgent
from agents.ifql import IFQLAgent
from agents.implicit_flows import ImplicitFlowsAgent
from agents.implicit_flows_v1 import ImplicitFlowsV1Agent
from agents.iql import IQLAgent
from agents.iqn import IQNAgent
from agents.rebrac import ReBRACAgent
from agents.sac import SACAgent
from agents.value_flows import ValueFlowsAgent

agents = dict(
    c51=C51Agent,
    codac=CODACAgent,
    fbrac=FBRACAgent,
    fql=FQLAgent,
    ifql=IFQLAgent,
    implicit_flows=ImplicitFlowsAgent,
    implicit_flows_v1=ImplicitFlowsV1Agent,
    iql=IQLAgent,
    iqn=IQNAgent,
    rebrac=ReBRACAgent,
    sac=SACAgent,
    value_flows=ValueFlowsAgent,
)
