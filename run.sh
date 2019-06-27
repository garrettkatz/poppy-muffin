docker run \
    --init \
    --rm \
    -it \
    --net=host \
    -e POPPY_LOCAL=$(getent hosts poppy.local | awk '{ print $1 }') \
    poppymuffin bash -c \
        "echo $POPPY_LOCAL && \
        mkdir /root/.ssh && \
        echo '$POPPY_LOCAL ecdsa-sha2-nistp256 AAAAE2VjZHNhLXNoYTItbmlzdHAyNTYAAAAIbmlzdHAyNTYAAABBBPolwwJWDyMxCJ+ibJWE/fAO0xQTC53sguEEtpryFAOTyBVf2BuszEJqVKMz0fG0MU5v1H+00ASK0FFGoenJWCM=' > /root/.ssh/known_hosts && \
        cat /root/.ssh/known_hosts \
        "

