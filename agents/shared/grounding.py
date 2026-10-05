"""Grounding metadata extraction and citation filtering for search agents."""

from models.agent import KeyPointWithCitation
from models.result import AgentStatus, GroundingSource
from pydantic import BaseModel


def extract_grounding_sources(raw) -> list[GroundingSource]:
    """Read titles and URIs from a Gemini grounding payload."""
    if raw is None:
        return []
    metadata = getattr(raw, "response_metadata", None) or {}
    if not isinstance(metadata, dict):
        return []
    grounding = metadata.get("grounding_metadata") or metadata.get("groundingMetadata") or {}
    if isinstance(grounding, dict):
        chunks = grounding.get("grounding_chunks") or grounding.get("groundingChunks") or []
    else:
        chunks = getattr(grounding, "grounding_chunks", None) or []

    sources: list[GroundingSource] = []
    for chunk in chunks:
        web = chunk.get("web") if isinstance(chunk, dict) else getattr(chunk, "web", None)
        if web is None:
            continue
        if isinstance(web, dict):
            title = web.get("title") or ""
            uri = web.get("uri") or ""
        else:
            title = getattr(web, "title", "") or ""
            uri = getattr(web, "uri", "") or ""
        if title or uri:
            sources.append(GroundingSource(title=title, uri=uri))
    return sources


def _source_matches(source: str, titles: list[str], uris: list[str]) -> bool:
    if not source:
        return False
    if source in titles or source in uris:
        return True
    for title in titles:
        if len(title) >= 4 and (title in source or source in title):
            return True
    for uri in uris:
        if uri and (uri in source or source in uri):
            return True
    return False


def filter_ungrounded_points(output: BaseModel, sources: list[GroundingSource]):
    """Drop citation points that do not match a grounding chunk.

    Returns the updated model and the number of points removed.
    When grounding returned no chunks, every cited point is removed.
    """
    points = list(getattr(output, "key_points", []) or [])
    if not points or not isinstance(points[0], KeyPointWithCitation):
        return output, 0

    titles = [source.title.casefold().strip() for source in sources if source.title]
    uris = [source.uri.casefold().strip() for source in sources if source.uri]
    if not titles and not uris:
        updated = output.model_copy(update={"key_points": []})
        return updated, len(points)

    kept = []
    for point in points:
        if _source_matches(point.source.casefold().strip(), titles, uris):
            kept.append(point)
    dropped = len(points) - len(kept)
    return output.model_copy(update={"key_points": kept}), dropped


def apply_grounding_filter(call):
    """Return a copy of an LLM call with ungrounded citations removed."""
    result = call.result
    if not isinstance(result, BaseModel):
        return call
    points = getattr(result, "key_points", None)
    if not points or not isinstance(points[0], KeyPointWithCitation):
        return call

    filtered, dropped = filter_ungrounded_points(result, call.grounding_sources)
    if dropped == 0:
        return call._replace(result=filtered)

    note = "Dropped key points that did not match search grounding sources."
    error = f"{call.error} {note}".strip() if call.error else note
    status = call.status if call.status == AgentStatus.FAILED else AgentStatus.DEGRADED
    return call._replace(result=filtered, status=status, error=error)
