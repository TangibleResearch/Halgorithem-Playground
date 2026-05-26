from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt

from .services import generate_chatgpt_response, parse_json_request, verify_payload


def index(request):
    return render(request, "verifier/index.html")


def health(request):
    return JsonResponse({"ok": True, "service": "halgo2-django"})


@csrf_exempt
def chatgpt(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    try:
        payload = parse_json_request(request)
        text = generate_chatgpt_response(
            payload.get("prompt", ""),
            model=payload.get("model"),
            api_key=payload.get("api_key"),
        )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"error": "ChatGPT request failed.", "detail": str(exc)}, status=502)
    return JsonResponse({"response_text": text})


@csrf_exempt
def verify(request):
    if request.method != "POST":
        return JsonResponse({"error": "POST required."}, status=405)
    try:
        if request.content_type and request.content_type.startswith("multipart/form-data"):
            data = request.POST
            response = verify_payload(
                source_text=data.get("source_text", ""),
                response_text=data.get("response_text", ""),
                source_name=data.get("source_name", "pasted_source"),
                urls=data.get("source_urls", ""),
                files=request.FILES.getlist("files"),
                threshold=data.get("threshold", 0.30),
            )
        else:
            payload = parse_json_request(request)
            response = verify_payload(
                source_text=payload.get("source_text", ""),
                response_text=payload.get("response_text", ""),
                source_name=payload.get("source_name", "pasted_source"),
                urls=payload.get("source_urls", []),
                threshold=payload.get("threshold", 0.30),
            )
    except ValueError as exc:
        return JsonResponse({"error": str(exc)}, status=400)
    except Exception as exc:
        return JsonResponse({"error": "Verification failed.", "detail": str(exc)}, status=500)
    return JsonResponse(response)
