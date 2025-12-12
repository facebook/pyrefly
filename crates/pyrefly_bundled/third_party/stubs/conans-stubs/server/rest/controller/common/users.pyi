from conans.errors import AuthenticationException as AuthenticationException
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.service.user_service import UserService as UserService

class UsersController:
    def attach_to(self, app): ...
