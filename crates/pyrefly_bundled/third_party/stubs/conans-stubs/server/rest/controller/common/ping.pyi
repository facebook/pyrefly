from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes

class PingController:
    @staticmethod
    def attach_to(app) -> None: ...
