from conans.model.ref import ConanFileReference as ConanFileReference
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.rest.controller.v2 import get_package_ref as get_package_ref
from conans.server.service.v2.service_v2 import ConanServiceV2 as ConanServiceV2

class RevisionsController:
    @staticmethod
    def attach_to(app): ...

def _format_rev_return(rev): ...
def _format_revs_return(revs): ...
