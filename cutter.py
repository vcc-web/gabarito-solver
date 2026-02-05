import cv2
import numpy as np
import os
import sys
import argparse


# --- FUNÇÕES AUXILIARES ---

def ler_imagem(path):
    """Lê imagem com suporte a nomes Unicode (Windows)."""
    stream = np.fromfile(path, dtype=np.uint8)
    img = cv2.imdecode(stream, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Não foi possível ler a imagem: {path}")
    return img


def salvar_imagem(path, img):
    """Salva imagem com suporte a nomes Unicode (Windows)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ['.jpg', '.jpeg']:
        params = [int(cv2.IMWRITE_JPEG_QUALITY), 95]
        _, buf = cv2.imencode('.jpg', img, params)
    elif ext == '.png':
        _, buf = cv2.imencode('.png', img)
    else:
        _, buf = cv2.imencode('.jpg', img)
    with open(path, 'wb') as f:
        f.write(buf.tobytes())


def redimensionar(image, altura_fixa=2000):
    """Redimensiona mantendo proporção para altura fixa."""
    h, w = image.shape[:2]
    ratio = altura_fixa / float(h)
    dim = (int(w * ratio), altura_fixa)
    return cv2.resize(image, dim, interpolation=cv2.INTER_AREA), ratio


def ordenar_cantos(pts):
    """
    Ordena 4 pontos na ordem: top-left, top-right, bottom-right, bottom-left.
    Usa soma (x+y) para achar TL/BR e diferença (x-y) para achar TR/BL.
    """
    pts = pts.reshape(4, 2).astype(np.float32)
    s = pts.sum(axis=1)
    d = np.diff(pts, axis=1).flatten()

    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(d)]
    bl = pts[np.argmax(d)]

    return np.array([tl, tr, br, bl], dtype=np.float32)


# --- DETECÇÃO DE COLUNAS ---

def detectar_colunas(image, min_area=50000, min_aspect=2.0):
    """
    Detecta os retângulos das colunas de respostas na imagem.

    Pipeline:
      1. Grayscale + GaussianBlur
      2. Canny edge detection
      3. Dilatação para fechar gaps nas bordas
      4. findContours + approxPolyDP para achar retângulos (4 vértices)
      5. Filtra por aspect ratio (alto > largo)

    Retorna lista de contornos (4 vértices cada), ordenados da esquerda
    para a direita.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecção de bordas
    edges = cv2.Canny(blur, 50, 150)

    # Dilata para fechar gaps nas bordas dos retângulos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Encontra contornos
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    colunas = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue

        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)

        # Precisa ter exatamente 4 vértices (retângulo)
        if len(approx) != 4:
            continue

        x, y, w, h = cv2.boundingRect(c)
        aspect = h / max(w, 1)

        # Filtro: colunas de gabarito são mais altas que largas
        if aspect > min_aspect:
            colunas.append(approx)

    # Ordena da esquerda para a direita
    colunas.sort(key=lambda c: cv2.boundingRect(c)[0])

    return colunas


# --- TRANSFORMAÇÃO DE PERSPECTIVA ---

def warp_retangulo(image, contorno):
    """
    Aplica transformação de perspectiva para "planificar" um retângulo
    detectado na imagem, corrigindo rotação e inclinação de uma vez.

    Diferente de warpAffine (só corrige rotação no plano), warpPerspective
    corrige distorções de perspectiva causadas pela câmera em ângulo.

    Args:
        image:     imagem original (full resolution)
        contorno:  contorno com 4 vértices (da detecção)

    Returns:
        Imagem retificada (sem recorte de margens), ou None se falhar
    """
    # Ordena os 4 cantos
    ordered = ordenar_cantos(contorno)
    tl, tr, br, bl = ordered

    # Calcula dimensões do retângulo de saída
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxW = int(max(widthA, widthB))

    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxH = int(max(heightA, heightB))

    if maxW == 0 or maxH == 0:
        return None

    # Retângulo destino alinhado
    dst = np.array([
        [0, 0],
        [maxW - 1, 0],
        [maxW - 1, maxH - 1],
        [0, maxH - 1]
    ], dtype=np.float32)

    # Calcula e aplica a matriz de perspectiva
    M = cv2.getPerspectiveTransform(ordered, dst)
    warped = cv2.warpPerspective(
        image, M, (maxW, maxH),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )

    return warped


def recortar_por_conteudo(warped, padding=5):
    """
    Recorta a imagem retificada ao redor do conteúdo real (bolinhas),
    usando detecção de círculos (HoughCircles) para encontrar os limites
    exatos da área de respostas.

    Isso é muito mais robusto que margens fixas percentuais, que podem
    cortar bolinhas nas bordas.

    Args:
        warped:   imagem já retificada por warp_retangulo
        padding:  pixels extras de margem ao redor dos círculos detectados

    Returns:
        Imagem recortada ao redor do conteúdo, ou a imagem original se
        não detectar círculos (fallback seguro).
    """
    h, w = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecta círculos (bolinhas de resposta) na imagem retificada
    circles = cv2.HoughCircles(
        blur,
        cv2.HOUGH_GRADIENT,
        dp=1,
        minDist=15,
        param1=50,
        param2=25,
        minRadius=8,
        maxRadius=30
    )

    if circles is None or len(circles[0]) < 20:
        # Fallback: se não detectar círculos suficientes, não recorta
        print("    (fallback: sem recorte adaptativo — poucos círculos detectados)")
        return warped

    circles = np.uint16(np.around(circles))
    all_c = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
    avg_r = int(np.mean([c[2] for c in all_c]))

    # Bounding box dos círculos + margem baseada no raio médio
    pad = avg_r + padding
    x1 = max(0, min(c[0] for c in all_c) - pad)
    y1 = max(0, min(c[1] for c in all_c) - pad)
    x2 = min(w, max(c[0] for c in all_c) + pad)
    y2 = min(h, max(c[1] for c in all_c) + pad)

    return warped[y1:y2, x1:x2]


# --- PIPELINE PRINCIPAL ---

def processar_gabarito(caminho_imagem, indice_coluna=0, debug=False, debug_dir=None):
    """
    Pipeline completo: lê a foto original do gabarito, detecta as colunas
    de respostas, corrige perspectiva e retorna a coluna recortada.

    Etapas:
      1. Lê imagem original (full res)
      2. Redimensiona para 2000px de altura (só para detecção)
      3. Detecta retângulos de colunas via Canny + contornos
      4. Escala os cantos de volta para resolução original
      5. Aplica getPerspectiveTransform + warpPerspective na full res
      6. Corta margens (bordas do retângulo)

    Args:
        caminho_imagem: caminho da foto original
        indice_coluna:  qual coluna extrair (0 = primeira/esquerda)
        debug:          se True, salva imagens de debug
        debug_dir:      diretório para salvar debug (None = mesmo dir da imagem)

    Returns:
        Imagem recortada da coluna, ou None se falhar
    """
    # 1. Ler imagem original
    img_original = ler_imagem(caminho_imagem)
    h_orig, w_orig = img_original.shape[:2]
    print(f"  Imagem original: {w_orig}x{h_orig}")

    # 2. Redimensionar para detecção (altura fixa = 2000px)
    ALTURA_DETECCAO = 2000
    img_pequena, ratio = redimensionar(img_original, ALTURA_DETECCAO)

    # 3. Detectar colunas na imagem reduzida
    colunas = detectar_colunas(img_pequena)
    print(f"  Colunas detectadas: {len(colunas)}")

    if not colunas:
        print("  ERRO: Nenhuma coluna de respostas encontrada!")
        return None

    if indice_coluna >= len(colunas):
        print(f"  ERRO: Coluna {indice_coluna + 1} não existe (total: {len(colunas)})")
        return None

    # 4. Escalar os cantos de volta para resolução original
    contorno = colunas[indice_coluna]
    contorno_fullres = (contorno.astype(np.float32) / ratio).astype(np.float32)

    # 5. Aplicar transformação de perspectiva na imagem ORIGINAL (full res)
    warped = warp_retangulo(img_original, contorno_fullres)

    if warped is None:
        print("  ERRO: Falha na transformação de perspectiva!")
        return None

    # 6. Recortar ao redor do conteúdo real (bolinhas) em vez de margens fixas
    resultado = recortar_por_conteudo(warped)

    print(f"  Resultado: {resultado.shape[1]}x{resultado.shape[0]}")

    # 6. Debug: salvar visualizações
    if debug:
        if debug_dir is None:
            debug_dir = os.path.dirname(caminho_imagem) or '.'
        debug_path = os.path.join(debug_dir, 'debug_output')
        os.makedirs(debug_path, exist_ok=True)

        nome_base = os.path.splitext(os.path.basename(caminho_imagem))[0]

        # Imagem com retângulos detectados
        img_debug = img_pequena.copy()
        for i, col in enumerate(colunas):
            cor = (0, 255, 0) if i == indice_coluna else (200, 200, 200)
            espessura = 3 if i == indice_coluna else 1
            cv2.drawContours(img_debug, [col], -1, cor, espessura)

            x, y, _, _ = cv2.boundingRect(col)
            cv2.putText(img_debug, f"#{i + 1}", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, cor, 2)

        debug_contornos = os.path.join(debug_path, f"{nome_base}_contornos.jpg")
        salvar_imagem(debug_contornos, img_debug)
        print(f"  Debug: {debug_contornos}")

        debug_resultado = os.path.join(debug_path, f"{nome_base}_coluna{indice_coluna + 1}.jpg")
        salvar_imagem(debug_resultado, resultado)
        print(f"  Debug: {debug_resultado}")

    return resultado


def processar_pasta(pasta_backup, pasta_saida, indice_coluna=0, debug=False):
    """
    Processa todas as imagens de uma pasta de backup.

    Args:
        pasta_backup:  pasta com as fotos originais (ex: student_exams/6o/backup)
        pasta_saida:   pasta onde salvar os recortes (ex: student_exams/6o)
        indice_coluna: qual coluna extrair (0 = primeira)
        debug:         salvar imagens de debug
    """
    if not os.path.isdir(pasta_backup):
        print(f"Pasta não encontrada: {pasta_backup}")
        return

    os.makedirs(pasta_saida, exist_ok=True)

    extensoes = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    arquivos = sorted([
        f for f in os.listdir(pasta_backup)
        if os.path.splitext(f)[1].lower() in extensoes
    ])

    print(f"Processando {len(arquivos)} imagens de '{pasta_backup}'...")
    print(f"Coluna alvo: #{indice_coluna + 1}")
    print("-" * 60)

    sucesso = 0
    falha = 0

    for arquivo in arquivos:
        caminho_entrada = os.path.join(pasta_backup, arquivo)
        caminho_saida = os.path.join(pasta_saida, arquivo)

        print(f"\n[{arquivo}]")

        try:
            resultado = processar_gabarito(
                caminho_entrada,
                indice_coluna=indice_coluna,
                debug=debug,
                debug_dir=pasta_saida
            )

            if resultado is not None:
                salvar_imagem(caminho_saida, resultado)
                print(f"  Salvo: {caminho_saida}")
                sucesso += 1
            else:
                falha += 1
        except Exception as e:
            print(f"  Erro: {e}")
            falha += 1

    print("\n" + "=" * 60)
    print(f"Concluído: {sucesso} OK, {falha} falhas, {len(arquivos)} total")


# --- EXECUÇÃO ---

def main():
    parser = argparse.ArgumentParser(
        description="Recorta e corrige perspectiva de gabaritos escaneados/fotografados."
    )
    parser.add_argument(
        "entrada",
        help="Caminho de uma imagem OU pasta 'backup' com imagens originais"
    )
    parser.add_argument(
        "-o", "--saida",
        default=None,
        help="Pasta de saída (padrão: pasta pai da entrada, ou '.' para arquivo único)"
    )
    parser.add_argument(
        "-c", "--coluna",
        type=int, default=1,
        help="Qual coluna extrair, 1-indexada (padrão: 1 = esquerda)"
    )
    parser.add_argument(
        "-d", "--debug",
        action="store_true",
        help="Salvar imagens de debug com contornos detectados"
    )

    args = parser.parse_args()

    indice_coluna = args.coluna - 1  # Converter para 0-indexado

    if os.path.isdir(args.entrada):
        # Modo pasta: processar todas as imagens
        pasta_backup = args.entrada
        pasta_saida = args.saida or os.path.dirname(pasta_backup.rstrip('/\\'))
        processar_pasta(pasta_backup, pasta_saida, indice_coluna, args.debug)

    elif os.path.isfile(args.entrada):
        # Modo arquivo único
        print(f"Processando: {args.entrada}")
        resultado = processar_gabarito(
            args.entrada,
            indice_coluna=indice_coluna,
            debug=args.debug,
            debug_dir=args.saida
        )

        if resultado is not None:
            if args.saida and os.path.isdir(args.saida):
                nome = os.path.basename(args.entrada)
                caminho_saida = os.path.join(args.saida, nome)
            elif args.saida:
                caminho_saida = args.saida
            else:
                nome = os.path.splitext(os.path.basename(args.entrada))[0]
                caminho_saida = f"{nome}_recortado.jpg"

            salvar_imagem(caminho_saida, resultado)
            print(f"Salvo: {caminho_saida}")

            # Mostrar resultado
            cv2.namedWindow("Resultado", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Resultado", 400, 800)
            cv2.imshow("Resultado", resultado)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Falha no processamento.")
            sys.exit(1)
    else:
        print(f"Entrada não encontrada: {args.entrada}")
        sys.exit(1)


if __name__ == "__main__":
    main()