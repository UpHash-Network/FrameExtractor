import os
import cv2
import gradio as gr
import numpy as np
from PIL import Image
from datetime import datetime
import shutil
import tempfile
import time
import threading

# グローバル変数として処理停止フラグとプレビュー画像を追加
stop_processing = False
current_preview_images = []

def reset_stop_flag():
    """処理停止フラグをリセットする"""
    global stop_processing, current_preview_images
    stop_processing = False
    current_preview_images = []

def set_stop_flag():
    """処理停止フラグを設定し、プレビュー画像を返す"""
    global stop_processing
    stop_processing = True
    return get_preview_images()

def is_video_file(filename):
    """動画ファイルかどうかを判定する"""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm')
    return filename.lower().endswith(video_extensions)

def find_video_files(directory):
    """指定されたディレクトリ内の動画ファイルを再帰的に検索する"""
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if is_video_file(file):
                video_files.append(os.path.join(root, file))
    return video_files

def save_frame(frame, output_file, output_format, quality):
    """フレームを画像として保存する"""
    try:
        # 出力ディレクトリの作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 一時ファイルの作成（delete=Falseで作成）
        temp_dir = tempfile.gettempdir()
        temp_filename = f"temp_frame_{os.path.basename(output_file)}"
        temp_path = os.path.join(temp_dir, temp_filename)
        
        # 画像を保存
        if output_format == "jpg":
            success = cv2.imwrite(temp_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        else:
            success = cv2.imwrite(temp_path, frame)
        
        if not success:
            print(f"警告: 一時ファイルへの保存に失敗しました: {temp_path}")
            return False
        
        # 一時ファイルを目的の場所に移動
        try:
            # 移動前に少し待機してファイルの解放を待つ
            time.sleep(0.1)
            
            # 移動を試みる
            shutil.move(temp_path, output_file)
            print(f"フレームを保存: {output_file}")
            return True
        except Exception as e:
            print(f"警告: ファイルの移動に失敗しました: {str(e)}")
            # 一時ファイルの削除を試みる
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except:
                pass
            return False
            
    except Exception as e:
        print(f"エラー: フレームの保存中にエラーが発生しました: {str(e)}")
        # 一時ファイルの削除を試みる
        try:
            if os.path.exists(temp_path):
                os.unlink(temp_path)
        except:
            pass
        return False

def calculate_sharpness(image):
    """画像の鮮明度を計算する（ラプラシアン分散を使用）"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()

def calculate_brightness_contrast(image):
    """画像の明度とコントラストを計算する"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray)
    contrast = np.std(gray)
    return mean_brightness, contrast

def is_good_quality_frame(image, sharpness_threshold=100, brightness_range=(50, 200), contrast_threshold=20):
    """フレームが良い品質かどうかを判定する"""
    sharpness = calculate_sharpness(image)
    brightness, contrast = calculate_brightness_contrast(image)
    
    # 鮮明度チェック
    if sharpness < sharpness_threshold:
        return False, f"低鮮明度: {sharpness:.2f}"
    
    # 明度チェック
    if not (brightness_range[0] <= brightness <= brightness_range[1]):
        return False, f"不適切な明度: {brightness:.2f}"
    
    # コントラストチェック
    if contrast < contrast_threshold:
        return False, f"低コントラスト: {contrast:.2f}"
    
    return True, f"良品質: 鮮明度={sharpness:.2f}, 明度={brightness:.2f}, コントラスト={contrast:.2f}"

def find_best_frame_in_range(cap, center_frame, search_range=5):
    """指定されたフレーム周辺で最も品質の良いフレームを見つける"""
    best_frame = None
    best_score = -1
    best_frame_index = center_frame
    
    # 検索範囲を設定
    start_frame = max(0, center_frame - search_range)
    end_frame = center_frame + search_range + 1
    
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 品質評価
        is_good, reason = is_good_quality_frame(frame)
        if is_good:
            sharpness = calculate_sharpness(frame)
            if sharpness > best_score:
                best_score = sharpness
                best_frame = frame.copy()
                best_frame_index = frame_idx
    
    # 元の位置に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    return best_frame, best_frame_index, best_score

def evaluate_frame_batch(cap, start_frame, end_frame, n_best=3):
    """指定された範囲のフレームを評価し、Best-Nを返す"""
    frame_scores = []
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, min(end_frame, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 品質評価
        sharpness = calculate_sharpness(frame)
        brightness, contrast = calculate_brightness_contrast(frame)
        
        # 総合スコアを計算（重み付き）
        # 鮮明度を最重要視し、明度とコントラストも考慮
        brightness_score = max(0, 100 - abs(brightness - 125))  # 125を理想値として
        contrast_score = min(contrast, 100)  # コントラストは100でキャップ
        
        total_score = (sharpness * 0.7) + (brightness_score * 0.2) + (contrast_score * 0.1)
        
        frame_scores.append({
            'frame_idx': frame_idx,
            'frame': frame.copy(),
            'sharpness': sharpness,
            'brightness': brightness,
            'contrast': contrast,
            'total_score': total_score
        })
    
    # 元の位置に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    # スコア順にソートして上位N個を返す
    frame_scores.sort(key=lambda x: x['total_score'], reverse=True)
    return frame_scores[:n_best]

def find_sharpest_frame_in_range(cap, target_frame, search_range, sharpness_threshold):
    """指定されたフレーム周辺で最も鮮明度が高く、閾値を超えるフレームを見つける"""
    best_frame = None
    best_sharpness = -1
    best_frame_index = target_frame
    
    # 検索範囲を設定
    start_frame = max(0, target_frame - search_range)
    end_frame = min(target_frame + search_range + 1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    original_position = cap.get(cv2.CAP_PROP_POS_FRAMES)
    
    for frame_idx in range(start_frame, end_frame):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        if not ret:
            continue
            
        # 鮮明度を計算
        sharpness = calculate_sharpness(frame)
        
        # 閾値を超えており、かつ現在の最高値より高い場合
        if sharpness >= sharpness_threshold and sharpness > best_sharpness:
            best_sharpness = sharpness
            best_frame = frame.copy()
            best_frame_index = frame_idx
    
    # 元の位置に戻す
    cap.set(cv2.CAP_PROP_POS_FRAMES, original_position)
    
    return best_frame, best_frame_index, best_sharpness

def extract_frames_from_video(
    video_path,
    frame_interval,
    output_format,
    quality,
    progress_func,
    video_index,
    total_videos,
    enable_quality_check=False,
    sharpness_threshold=100,
    search_range=5
):
    """単一の動画からフレームを抽出する"""
    global stop_processing
    
    if not os.path.exists(video_path):
        return None, "エラー: 動画ファイルが見つかりません。"
    
    # 動画ファイル名からディレクトリ名を生成
    video_dir = os.path.dirname(video_path)
    video_filename = os.path.splitext(os.path.basename(video_path))[0]
    frames_dir = f"{video_filename}_frames"
    output_path = os.path.join(video_dir, frames_dir)
    
    try:
        # 出力ディレクトリが存在する場合は削除
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
        os.makedirs(output_path, exist_ok=True)
        print(f"出力ディレクトリを作成: {output_path}")
    except Exception as e:
        return None, f"エラー: 出力ディレクトリの作成に失敗しました: {str(e)}"
    
    # 動画の読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None, f"エラー: 動画を開けませんでした: {video_path}"
    
    # 動画情報の取得
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"動画情報: {total_frames}フレーム, {fps}fps")
    
    # フレーム抽出
    saved_count = 0
    skipped_count = 0
    preview_images = []
    processed_count = 0
    
    # 指定間隔でフレームを処理
    for target_frame in range(0, total_frames, frame_interval):
        if stop_processing:
            cap.release()
            return preview_images, "処理が停止されました。"
        
        final_frame = None
        frame_info = ""
        actual_frame_index = target_frame
        
        if enable_quality_check:
            # 指定フレーム周辺で最も鮮明なフレームを探す
            best_frame, best_frame_index, best_sharpness = find_sharpest_frame_in_range(
                cap, target_frame, search_range, sharpness_threshold
            )
            
            if best_frame is not None:
                final_frame = best_frame
                actual_frame_index = best_frame_index
                if best_frame_index != target_frame:
                    frame_info = f" (最適化: フレーム{target_frame}→{best_frame_index}, 鮮明度:{best_sharpness:.2f})"
                else:
                    frame_info = f" (元フレーム使用, 鮮明度:{best_sharpness:.2f})"
            else:
                # 閾値を超えるフレームが見つからない場合はスキップ
                skipped_count += 1
                print(f"フレーム{target_frame}周辺: 閾値({sharpness_threshold})を超える鮮明なフレームが見つかりませんでした")
        else:
            # 品質チェックが無効の場合は指定フレームをそのまま使用
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
            ret, frame = cap.read()
            
            if ret:
                final_frame = frame
                sharpness = calculate_sharpness(frame)
                frame_info = f" (鮮明度:{sharpness:.2f})"
            else:
                skipped_count += 1
        
        if final_frame is not None:
            # BGRからRGBに変換
            frame_rgb = cv2.cvtColor(final_frame, cv2.COLOR_BGR2RGB)
            
            # 画像の保存
            output_file = os.path.join(output_path, f"frame_{saved_count:06d}.{output_format}")
            if save_frame(final_frame, output_file, output_format, quality):
                # プレビュー用に画像を保存（最初の3枚のみ）
                if len(preview_images) < 3:
                    preview_images.append(frame_rgb)
                saved_count += 1
                print(f"フレーム保存: {output_file}{frame_info}")
        
        processed_count += 1
        
        # プログレス更新（10フレームごと、またはスキップされた場合も含む）
        if processed_count % 10 == 0 or final_frame is None:
            frame_progress = target_frame / total_frames
            overall_progress = (video_index + frame_progress) / total_videos
            desc = f"動画 {video_index + 1}/{total_videos}: {os.path.basename(video_path)} (進捗 {target_frame}/{total_frames})"
            if enable_quality_check:
                desc += f" [保存:{saved_count}, スキップ:{skipped_count}]"
            progress_func(overall_progress, desc=desc, total=None, unit=None)
    
    cap.release()
    
    # 結果の表示
    result_text = f"""
    処理が完了しました！
    動画: {video_path}
    保存されたフレーム数: {saved_count}
    """
    if enable_quality_check:
        result_text += f"    スキップされたフレーム数: {skipped_count}\n"
        result_text += f"    鮮明度閾値: {sharpness_threshold}\n"
        result_text += f"    検索範囲: ±{search_range}フレーム\n"
    result_text += f"    出力ディレクトリ: {output_path}"
    
    return preview_images, result_text

def process_directory(
    input_dir,
    frame_interval,
    output_format,
    quality,
    enable_quality_check,
    sharpness_threshold,
    search_range,
    progress=gr.Progress()
):
    """ディレクトリ内の動画を処理する"""
    global stop_processing, current_preview_images
    reset_stop_flag()  # 処理開始時にフラグをリセット
    
    if not os.path.exists(input_dir):
        return "エラー: 指定されたディレクトリが見つかりません。"
    
    print(f"処理開始: {input_dir}")
    
    # 初期プログレス表示
    progress(0, desc="動画ファイルを検索中...")
    
    video_files = find_video_files(input_dir)
    if not video_files:
        return "エラー: 指定されたディレクトリ内に動画ファイルが見つかりません。"
    
    print(f"見つかった動画ファイル: {len(video_files)}個")
    for video in video_files:
        print(f"- {video}")
    
    all_results = []
    total_videos = len(video_files)
    
    # 各動画ファイルを処理
    for video_index, video_path in enumerate(video_files):
        if stop_processing:
            final_result = "\n\n".join(all_results) + "\n\n処理が停止されました。"
            return final_result
        
        # 現在の動画の処理状況を表示
        previews, result = extract_frames_from_video(
            video_path,
            frame_interval,
            output_format,
            quality,
            progress,
            video_index,
            total_videos,
            enable_quality_check,
            sharpness_threshold,
            search_range
        )
        
        if previews:
            current_preview_images.extend(previews)
        all_results.append(result)
        
        if stop_processing:
            break
    
    # 完了メッセージ
    progress(1.0, desc="全ての処理が完了しました！")
    
    return "\n\n".join(all_results)  # テキストのみ返す

def get_preview_images():
    """停止時や完了時にプレビュー画像を取得する"""
    global current_preview_images
    return current_preview_images[:3]  # 最初の3枚のみ返す

def on_process_complete():
    """処理完了時にプレビュー画像を更新する"""
    return get_preview_images()

# Gradioインターフェースの作成
with gr.Blocks(title="動画フレーム抽出ツール") as demo:
    gr.Markdown("# 動画フレーム抽出ツール")
    gr.Markdown("複数の動画ファイルからフレームを抽出し、個別のフォルダに保存します。")
    
    with gr.Row():
        with gr.Column():
            input_dir = gr.Textbox(
                label="入力ディレクトリ",
                placeholder="処理する動画が含まれるディレクトリのパスを入力してください",
                interactive=True
            )
            
            with gr.Row():
                frame_interval = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=30,
                    step=1,
                    label="フレーム間隔（何フレームごとに抽出するか）",
                    info="この間隔で指定されたフレームの前後から最適なフレームを選択します"
                )
                quality = gr.Slider(
                    minimum=1,
                    maximum=100,
                    value=95,
                    step=1,
                    label="JPEG品質（jpg形式の場合）"
                )
            
            output_format = gr.Radio(
                choices=["jpg", "png"],
                value="png",
                label="出力形式"
            )
            
            # 品質チェック機能の設定
            with gr.Accordion("鮮明度ベース選択設定", open=True):
                enable_quality_check = gr.Checkbox(
                    label="鮮明度ベース選択を有効にする",
                    value=True,
                    info="指定間隔のフレーム前後から最も鮮明で閾値を超えるフレームを選択します"
                )
                
                with gr.Row():
                    sharpness_threshold = gr.Slider(
                        minimum=50,
                        maximum=500,
                        value=100,
                        step=10,
                        label="鮮明度閾値",
                        info="この値以上の鮮明度を持つフレームのみを選択します"
                    )
                    search_range = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="検索範囲",
                        info="指定フレームの前後何フレームまで検索するか"
                    )
                
                gr.Markdown("""
                **鮮明度ベース選択の動作:**
                1. フレーム間隔で指定されたフレーム（例：30フレームごと）を基準点とする
                2. その基準点の前後±検索範囲内のフレームを全て評価
                3. 鮮明度閾値を超えるフレームの中から最も鮮明度が高いものを選択
                4. 閾値を超えるフレームがない場合はスキップ
                
                **無効にした場合:** 指定間隔のフレームをそのまま保存
                """)
            
            with gr.Row():
                extract_button = gr.Button("フレーム抽出開始", variant="primary")
                stop_button = gr.Button("処理停止", variant="stop")
        
        with gr.Column():
            result_text = gr.Textbox(
                label="処理結果", 
                lines=10,
                max_lines=15,
                show_copy_button=True
            )
            preview_gallery = gr.Gallery(
                label="プレビュー（最初の3枚）",
                show_label=True,
                columns=3,
                height=300,
                object_fit="cover"
            )
    
    # イベントハンドラー
    extract_button.click(
        fn=process_directory,
        inputs=[
            input_dir,
            frame_interval,
            output_format,
            quality,
            enable_quality_check,
            sharpness_threshold,
            search_range
        ],
        outputs=[result_text]
    ).then(
        fn=on_process_complete,
        inputs=[],
        outputs=[preview_gallery]
    )
    
    stop_button.click(
        fn=set_stop_flag,
        inputs=[],
        outputs=[preview_gallery]
    )

if __name__ == "__main__":
    demo.launch()