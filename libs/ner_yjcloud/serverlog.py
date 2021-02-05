"""
Usage: Server log support for chinese whereas ASCII

This method replace Default NLU logger and used in `data_router.py`

"""

from twisted.logger._json import *

def nonJsonLogger(event):
    """
    @param event: A log event dictionary.
    @type event: L{dict} with arbitrary keys and values

    @return: A string of the serialized JSON; note that this will contain no
        newline characters, and may thus safely be stored in a line-delimited
        file.
    @rtype: L{unicode}
    """
    if bytes is str:
        kw = dict(default=objectSaveHook, encoding="charmap", skipkeys=True)
    else:
        def default(unencodable):
            """
            Serialize an object not otherwise serializable by L{dumps}.

            @param unencodable: An unencodable object.
            @return: C{unencodable}, serialized
            """
            if isinstance(unencodable, bytes):
                return unencodable.decode("charmap")
            return objectSaveHook(unencodable)


        kw = dict(default=default, skipkeys=True)

    flattenEvent(event)
    result = dumps(event, **kw, ensure_ascii = False)
    if not isinstance(result, unicode):
        return unicode(result, "utf-8", "replace")
    return result


def nonJsonFileLogObserver(outFile, recordSeparator=u"\x1e"):
    """
    Create a L{FileLogObserver} that emits Non-Json events to a
    specified (writable) file-like object.

    @param outFile: A file-like object.  Ideally one should be passed which
        accepts L{unicode} data.  Otherwise, UTF-8 L{bytes} will be used.
    @type outFile: L{io.IOBase}

    @param recordSeparator: The record separator to use.
    @type recordSeparator: L{unicode}

    @return: A file log observer.
    @rtype: L{FileLogObserver}
    """
    return FileLogObserver(
        outFile,
        lambda event: u"{0}{1}\n".format(recordSeparator, nonJsonLogger(event))
    )
