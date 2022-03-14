# symlink support under windows:
# don't symlink but instead just copy
import os
import platform

if platform.system() == 'Windows':
    def symlink_ms(source, link_name):
        os.system("copy {target} {link}".format(
            link=link_name,
            target=source.replace('/', '\\')))

    os.symlink = symlink_ms
