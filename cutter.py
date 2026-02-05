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

def detectar_colunas(image, min_area=30000, min_aspect=2.0, expected_cols=5):
    """
    Detecta os retângulos das colunas de respostas na imagem.

    Pipeline com múltiplas camadas de detecção:
      1. Grayscale + GaussianBlur + Canny + dilatação
      2. findContours (RETR_LIST) para não perder contornos internos
      3. Filtra por aspect ratio, fill ratio e área mínima
      4. De-duplica agrupando por center_x (tolerância 80px)
      5. Se < expected_cols: usa cabeçalhos das colunas (retângulos baixos
         e largos acima de cada coluna) para interpolar posições ausentes,
         construindo retângulos sintéticos.

    Retorna lista de contornos (4 pontos cada), ordenados da esquerda
    para a direita.
    """
    h_img, w_img = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detecção de bordas
    edges = cv2.Canny(blur, 50, 150)

    # Dilata para fechar gaps nas bordas dos retângulos
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Encontra contornos (RETR_LIST para manter contornos internos
    # quando bordas de colunas adjacentes se tocam)
    cnts, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # --- Camada 1: detectar colunas diretamente ---
    column_candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / max(w, 1)
        if aspect < min_aspect:
            continue
        bbox_area = w * h
        fill = area / bbox_area if bbox_area > 0 else 0
        # fill_ratio > 0.5 garante que é um retângulo razoável,
        # não apenas um contorno de borda (que teria fill ~0.25)
        if fill < 0.5:
            continue
        cx = x + w // 2
        column_candidates.append((cx, area, c, x, y, w, h))

    # De-duplicar: agrupar por center_x com tolerância de 80px
    # (RETR_LIST gera contornos internos e externos para a mesma coluna)
    column_candidates.sort(key=lambda item: item[0])
    groups = []
    for item in column_candidates:
        cx = item[0]
        placed = False
        for g in groups:
            g_mean_cx = np.mean([i[0] for i in g])
            if abs(cx - g_mean_cx) < 80:
                g.append(item)
                placed = True
                break
        if not placed:
            groups.append([item])

    # Manter o contorno de maior área em cada grupo
    detected_cols = []
    for g in groups:
        best = max(g, key=lambda item: item[1])
        detected_cols.append(best)

    detected_cols.sort(key=lambda item: item[0])

    # Se temos o esperado, retornar direto
    if len(detected_cols) >= expected_cols:
        cols = [item[2] for item in detected_cols[:expected_cols]]
        cols.sort(key=lambda c: cv2.boundingRect(c)[0])
        return cols

    # --- Camada 2: interpolar colunas ausentes via cabeçalhos ---
    # Os cabeçalhos são retângulos baixos e largos acima de cada coluna.
    # São SEMPRE detectados porque são pequenos e isolados.
    header_candidates = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 12000 or area > 50000:
            continue
        x, y, w, h = cv2.boundingRect(c)
        aspect = h / max(w, 1)
        # Cabeçalhos: mais largos que altos (aspect < 0.8)
        if aspect > 0.8 or aspect < 0.2:
            continue
        bbox_area = w * h
        fill = area / bbox_area if bbox_area > 0 else 0
        # Cabeçalhos são muito retangulares
        if fill < 0.80:
            continue
        # Largura razoável para cabeçalho (entre 10% e 30% da imagem)
        w_ratio = w / w_img
        if w_ratio < 0.10 or w_ratio > 0.30:
            continue
        cx = x + w // 2
        cy = y + h // 2
        header_candidates.append((cx, cy, x, y, w, h, area, c))

    # Agrupar cabeçalhos por y (mesma fila)
    # Usa média do grupo para tolerância, porque a folha pode estar
    # inclinada (~50px de variação entre cabeçalhos extremos)
    header_candidates.sort(key=lambda item: item[1])
    header_rows = []
    for item in header_candidates:
        cy = item[1]
        placed = False
        for row in header_rows:
            row_mean_cy = np.mean([i[1] for i in row])
            if abs(cy - row_mean_cy) < 60:
                row.append(item)
                placed = True
                break
        if not placed:
            header_rows.append([item])

    # De-duplicar cada fila por cx (tolerância 50px) e escolher a melhor fila
    best_row = None
    best_row_count = 0
    for row in header_rows:
        row.sort(key=lambda item: item[0])
        deduped = []
        for item in row:
            placed = False
            for g in deduped:
                if abs(item[0] - g[0]) < 50:
                    # Manter o de maior área
                    if item[6] > g[6]:
                        g[:] = list(item)
                    placed = True
                    break
            if not placed:
                deduped.append(list(item))

        if len(deduped) > best_row_count and len(deduped) >= 3:
            best_row = deduped
            best_row_count = len(deduped)

    if best_row is None or len(best_row) < 3:
        # Sem cabeçalhos suficientes, retornar o que temos
        return [item[2] for item in detected_cols]

    best_row.sort(key=lambda item: item[0])

    # Dimensões médias das colunas detectadas (para construir sintéticas)
    if detected_cols:
        avg_col_w = int(np.mean([item[5] for item in detected_cols]))
        avg_col_h = int(np.mean([item[6] for item in detected_cols]))
        avg_col_top = int(np.mean([item[4] for item in detected_cols]))
    else:
        # Estimar a partir dos cabeçalhos
        avg_col_w = int(np.mean([h[4] for h in best_row]))
        avg_col_h = int(avg_col_w * 4)
        avg_col_top = int(np.mean([h[3] + h[5] for h in best_row]))

    # Associar cada cabeçalho a uma coluna detectada ou criar sintética
    all_columns = []
    for hdr in best_row:
        hdr_cx = hdr[0]
        # Procurar coluna detectada mais próxima
        matched = None
        min_dist = 80
        for det in detected_cols:
            dist = abs(hdr_cx - det[0])
            if dist < min_dist:
                min_dist = dist
                matched = det

        if matched is not None:
            all_columns.append(matched[2])
        else:
            # Construir retângulo sintético alinhado com o cabeçalho
            x = hdr[2]
            y = avg_col_top
            w = avg_col_w
            h = avg_col_h
            # Garantir que não ultrapasse a imagem
            x = max(0, x)
            y = max(0, y)
            if x + w > w_img:
                w = w_img - x
            if y + h > h_img:
                h = h_img - y
            synthetic = np.array([
                [[x, y]],
                [[x + w, y]],
                [[x + w, y + h]],
                [[x, y + h]]
            ], dtype=np.int32)
            all_columns.append(synthetic)

    all_columns.sort(key=lambda c: cv2.boundingRect(c)[0])
    return all_columns


# --- TRANSFORMAÇÃO DE PERSPECTIVA ---

def warp_retangulo(image, contorno):
    """
    Aplica transformação de perspectiva para "planificar" um retângulo
    detectado na imagem, corrigindo rotação e inclinação de uma vez.

    Diferente de warpAffine (só corrige rotação no plano), warpPerspective
    corrige distorções de perspectiva causadas pela câmera em ângulo.

    Aceita contornos com qualquer número de vértices (>= 4). Para contornos
    com mais de 4 vértices, usa minAreaRect para extrair os 4 cantos do
    retângulo mínimo que engloba o contorno.

    Args:
        image:     imagem original (full resolution)
        contorno:  contorno com >= 4 vértices (da detecção)

    Returns:
        Imagem retificada (sem recorte de margens), ou None se falhar
    """
    # Normalizar contorno para (N, 2)
    pts = contorno.reshape(-1, 2).astype(np.float32)

    if len(pts) == 4:
        # Contorno já é um quadrilátero — usar diretamente
        ordered = ordenar_cantos(pts.reshape(4, 2))
    else:
        # Para contornos com mais vértices, usar minAreaRect
        # para obter o retângulo mínimo rotacionado
        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect)
        ordered = ordenar_cantos(box)

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