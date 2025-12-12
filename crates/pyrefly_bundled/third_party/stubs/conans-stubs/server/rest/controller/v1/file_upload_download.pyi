from bottle import FileUpload
from conans.server.rest.bottle_routes import BottleRoutes as BottleRoutes
from conans.server.service.mime import get_mime_type as get_mime_type
from conans.server.service.v1.upload_download_service import FileUploadDownloadService as FileUploadDownloadService

class FileUploadDownloadController:
    @staticmethod
    def attach_to(app): ...

class ConanFileUpload(FileUpload):
    def filename(self): ...
