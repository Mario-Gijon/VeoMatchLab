"""Local Chromium integration and CPU/GLSL parity check. No Veo network access.

Run from the repository root: .venv/bin/python tests/browser_check.py
Starts temporary servers and isolated match storage; never touches historical output.
"""
from __future__ import annotations

import base64
import copy
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from urllib.request import urlopen

import cv2
import numpy as np
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from geometry.reproduce_veo_panorama import render
from geometry.veo_metadata import load_veo_metadata
from web_app.backend.app.view_geometry import view_context
from geometry.tracking_ground_v3 import logic_matrices


def ready(url, process):
    for _ in range(100):
        if process.poll() is not None:
            raise RuntimeError("Test server exited before startup")
        try:
            with urlopen(url, timeout=1) as response:
                if response.status == 200:
                    return
        except OSError:
            time.sleep(.1)
    raise RuntimeError("Test server startup timed out")


def check_interactive_pixels(page, context, png, image, metadata):
    """Compare GPU perspective ground pixels to equivalent validated CPU rays."""
    result = page.evaluate('''async ({context, png}) => {
      const {createPanoramaGL} = await import('/src/components/panoramaGL.ts');
      const {initialCamera, interactiveTransform, interactiveGroundPoint} = await import('/src/components/interactiveCamera.ts');
      const canvas=document.createElement('canvas'); canvas.width=320; canvas.height=180;
      const renderer=createPanoramaGL(canvas,context,true);
      const image=new Image();image.src='data:image/png;base64,'+png;await image.decode();
      const camera=initialCamera(context);camera.longitude+=.15;camera.latitude+=.1;camera.fov=.8;
      renderer.draw(image,{matrix:interactiveTransform(context,camera),fov:camera.fov});
      const samples=[];
      for(const y of [30,70,110,150]) for(const x of [50,130,210,290]) {
        const point=interactiveGroundPoint(context,camera,x,y,320,180);
        if(point) samples.push({x,y,point});
      }
      const pngOut=canvas.toDataURL();renderer.dispose();return {png:pngOut,samples};
    }''', {'context': context, 'png': png})
    gpu = cv2.imdecode(np.frombuffer(base64.b64decode(result['png'].split(',')[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
    assert len(result['samples']) >= 4
    max_error = 0
    for sample in result['samples']:
        target = np.array([sample['point']['world_x'], 0, sample['point']['world_z']])
        ray = np.linalg.inv(metadata.camera2world @ np.linalg.inv(metadata.camera2presentation)) @ np.r_[target - metadata.camera2world[:3, 3], 0]
        lon = float(np.arctan2(ray[0], ray[2]))
        lat = float(-np.arcsin(ray[1] / np.linalg.norm(ray[:3])))
        single = copy.copy(metadata)
        single.va_presentation = {'left': lon, 'right': lon, 'top': lat, 'bottom': lat}
        with redirect_stdout(io.StringIO()):
            expected = render(image, single, 1, 1)['panorama'][0, 0]
        error = np.max(np.abs(expected.astype(float) - gpu[sample['y'], sample['x']].astype(float)))
        max_error = max(max_error, error)
        assert error < 8, f'Interactive GPU / validated CPU ray mismatch: {error}'
    print(f"Interactive GPU/CPU parity: {len(result['samples'])} ground pixels, max RGB error {max_error}", flush=True)


def main():
    with tempfile.TemporaryDirectory(prefix="veo-browser-check-") as temporary:
        work = Path(temporary)
        env = {**os.environ, "VEO_MATCH_DATA_ROOT": str(work / "runtime"), "VITE_API_BASE_URL": "http://127.0.0.1:8005", "VEO_FRONTEND_ORIGINS": "http://127.0.0.1:5175"}
        with (work / 'servers.log').open('w') as log:
            backend = subprocess.Popen([sys.executable, '-m', 'uvicorn', 'web_app.backend.app.main:app', '--host', '127.0.0.1', '--port', '8005'], cwd=ROOT, env=env, stdout=log, stderr=log)
            frontend = subprocess.Popen(['node', 'node_modules/vite/bin/vite.js', '--host', '127.0.0.1', '--port', '5175', '--strictPort'], cwd=ROOT / 'web_app/frontend', env=env, stdout=log, stderr=log)
            try:
                ready('http://127.0.0.1:8005/api/health', backend)
                with urlopen('http://127.0.0.1:8005/openapi.json') as response:
                    specification = json.load(response)
                    assert '/api/view-point' in specification['paths']
                    assert '/api/veo/inspect' in specification['paths']
                    print('FastAPI startup and OpenAPI passed', flush=True)
                ready('http://127.0.0.1:5175', frontend)
                with sync_playwright() as playwright:
                    browser = playwright.chromium.launch(headless=True, args=['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'])
                    page = browser.new_page(viewport={"width": 2200, "height": 1200})
                    errors = []
                    page.on('pageerror', lambda error: errors.append(str(error)))
                    page.goto('http://127.0.0.1:5175')
                    page.get_by_role('heading', name='Matches', exact=True).wait_for()
                    for index, calibration in enumerate(sorted((ROOT / 'tests/fixtures/calibrations').glob('*.veo'))):
                        image_path = ROOT / 'tests/fixtures/images' / ('stacked_frame.jpg' if index == 0 else 'match2_stacked_frame.jpg')
                        image = cv2.imread(str(image_path))
                        png = base64.b64encode(cv2.imencode('.png', image)[1]).decode()
                        context = view_context(str(calibration))
                        actual = page.evaluate('''async ({context, png}) => {
                          const {createPanoramaGL} = await import('/src/components/panoramaGL.ts');
                          const canvas = document.createElement('canvas'); canvas.width=640; canvas.height=360;
                          const renderer = createPanoramaGL(canvas, context);
                          const image = new Image(); image.src='data:image/png;base64,'+png; await image.decode();
                          renderer.draw(image); const result=canvas.toDataURL('image/png'); renderer.dispose();
                          const shared = createPanoramaGL(canvas, context, true);
                          shared.draw(image); const sharedResult=canvas.toDataURL('image/png'); shared.dispose();
                          if (result !== sharedResult) throw Error('Shared renderer changed Panorama pixels');
                          return result;
                        }''', {'context': context, 'png': png})
                        actual_image = cv2.imdecode(np.frombuffer(base64.b64decode(actual.split(',')[1]), dtype=np.uint8), cv2.IMREAD_COLOR)
                        expected = render(image, load_veo_metadata(calibration), 640, 360)['panorama']
                        difference = np.abs(expected.astype(float) - actual_image.astype(float))
                        mean, p99 = float(difference.mean()), float(np.percentile(difference, 99))
                        print(json.dumps({'match': calibration.name, 'mean_rgb_error': mean, 'p99_rgb_error': p99}), flush=True)
                        assert mean < 2 and p99 < 10, "Captured shader / CPU render parity failed"
                        check_interactive_pixels(page, context, png, image, load_veo_metadata(calibration))
                    # Generate a tiny-duration local fixture with browser-compatible codec.
                    clip = work / 'dual.mp4'
                    subprocess.run(['ffmpeg', '-v', 'error', '-loop', '1', '-i', str(ROOT / 'tests/fixtures/images/stacked_frame.jpg'), '-t', '3', '-r', '5', '-c:v', 'libx264', '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', str(clip)], check=True)
                    request = page.request
                    calibrations = request.get('http://127.0.0.1:8005/api/calibrations').json()
                    match = request.post('http://127.0.0.1:8005/api/matches', data={'name': 'Browser fixture', 'calibration_id': calibrations[0]['id']}).json()
                    for role in ['dual_lens', 'tracking']:
                        response = request.post(f"http://127.0.0.1:8005/api/matches/{match['id']}/media/path", data={'path': str(clip), 'role': role})
                        assert response.status == 201
                    metadata = load_veo_metadata(Path(calibrations[0]['path']))
                    _, logic2camera = logic_matrices(metadata)
                    center_ray = np.linalg.inv(metadata.camera2world @ logic2camera) @ np.array([
                        -metadata.camera2world[0, 3], -metadata.camera2world[1, 3], -metadata.camera2world[2, 3], 0])
                    yaw = np.arctan2(center_ray[0], center_ray[2])
                    pitch = np.arctan2(-center_ray[1], np.hypot(center_ray[0], center_ray[2]))
                    det = work / 'detections_constrained.det'
                    det.write_text(f'{yaw},{pitch},1,0\n' * 60)
                    response = request.post(f"http://127.0.0.1:8005/api/matches/{match['id']}/media/path", data={'path': str(det), 'role': 'camera_directions'})
                    assert response.status == 201
                    response = request.post(f"http://127.0.0.1:8005/api/matches/{match['id']}/media/path", data={'path': str(clip), 'role': 'veo_highlights'})
                    assert response.status == 201
                    page.reload()
                    captures = Path(os.environ['VEO_UI_CAPTURE_DIR']) if os.environ.get('VEO_UI_CAPTURE_DIR') else None
                    if captures:
                        captures.mkdir(parents=True, exist_ok=True)
                        page.get_by_role('button', name='Open match', exact=True).wait_for()
                        page.screenshot(path=str(captures / 'library.png'))
                    page.get_by_role('button', name='Open match', exact=True).click()
                    page.wait_for_function("document.querySelector('video')?.readyState >= 2")
                    page.get_by_label('Time (s)').fill('1.2')
                    page.wait_for_function("Math.abs(document.querySelector('video').currentTime-1.2)<0.02 && !document.querySelector('video').seeking")
                    # Enter Interactive on the same video element/clock, with a
                    # persistent WebGL context. No duplicate decoding pipeline.
                    page.evaluate("window.originalVideo = document.querySelector('video'); window.originalCanvas = document.querySelector('canvas[aria-label=\"Panorama view\"]')")
                    page.get_by_role('button', name='Interactive', exact=True).click()
                    interactive = page.locator('canvas[aria-label="Interactive view"]')
                    interactive.wait_for(state='visible')
                    page.wait_for_function("document.querySelector('canvas[aria-label=\"Interactive view\"]').width > 300")
                    assert page.evaluate("window.originalVideo === document.querySelector('video') && window.originalCanvas === document.querySelector('canvas[aria-label=\"Interactive view\"]')")
                    assert abs(page.locator('video').evaluate('(v) => v.currentTime') - 1.2) < .02
                    assert page.locator('video').count() == 1
                    page.get_by_role('button', name='View Veo highlights', exact=True).click()
                    page.get_by_role('dialog').wait_for()
                    assert page.get_by_role('dialog').locator('video[controls]').count() == 1
                    page.get_by_role('button', name='Close', exact=True).click()
                    # Find a deterministic in-field click using the production
                    # projection, then confirm the real pointer path selects it.
                    click = page.evaluate('''async (context) => {
                      const {initialCamera, interactiveGroundPoint} = await import('/src/components/interactiveCamera.ts');
                      const c = document.querySelector('canvas[aria-label="Interactive view"]');
                      const bounds = c.getBoundingClientRect();
                      for (let y=.2; y<.9; y+=.1) for(let x=.2; x<.9; x+=.1) {
                        const px=Math.floor(x*c.width), py=Math.floor(y*c.height);
                        const point=interactiveGroundPoint(context,initialCamera(context),px,py,c.width,c.height);
                        if(point) return {x:(px+.5)/c.width*bounds.width,y:(py+.5)/c.height*bounds.height,point};
                      }
                      throw Error('No field pixels in initial Interactive camera');
                    }''', view_context(str(calibrations[0]['path'])))
                    interactive.click(position={'x': click['x'], 'y': click['y']})
                    page.get_by_role('button', name='Measurements · 1 points').wait_for()
                    marker = page.locator('svg[aria-label="Pitch reference"] circle[fill="#ffcf4a"]')
                    assert marker.count() == 1
                    assert abs(float(marker.get_attribute('cx')) - click['point']['world_x']) < 1e-6
                    page.get_by_role('button', name='Select', exact=True).click()
                    assert page.get_by_label('Projected pitch marker').count() == 0
                    interactive.click(position={'x': click['x'], 'y': click['y']})
                    assert page.get_by_role('button', name='Measurements · 1 points').count() == 1
                    before = interactive.evaluate('(c) => c.toDataURL()')
                    box = interactive.bounding_box()
                    page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
                    page.mouse.down(); page.mouse.move(box['x'] + box['width']/2 + 40, box['y'] + box['height']/2, steps=4); page.mouse.up()
                    page.wait_for_function("old => document.querySelector('canvas[aria-label=\"Interactive view\"]').toDataURL() !== old", arg=before)
                    assert marker.count() == 1
                    page.get_by_role('button', name='Select', exact=True).click()
                    before = interactive.evaluate('(c) => c.toDataURL()')
                    box = interactive.bounding_box()
                    page.mouse.move(box['x'] + box['width']/2, box['y'] + box['height']/2)
                    page.mouse.down(); page.mouse.move(box['x'] + box['width']/2 + 60, box['y'] + box['height']/2 + 20, steps=8); page.mouse.up()
                    page.wait_for_function("old => document.querySelector('canvas[aria-label=\"Interactive view\"]').toDataURL() !== old", arg=before)
                    assert page.get_by_role('button', name='Measurements · 1 points').count() == 1
                    before = interactive.evaluate('(c) => c.toDataURL()')
                    page.mouse.wheel(0, -200)
                    page.wait_for_function("old => document.querySelector('canvas[aria-label=\"Interactive view\"]').toDataURL() !== old", arg=before)
                    selected = marker.get_attribute('cx')
                    page.get_by_role('button', name='Play', exact=True).click()
                    page.wait_for_function("document.querySelector('video').currentTime > 1.4")
                    page.mouse.wheel(0, 100)
                    page.get_by_role('button', name='Panorama', exact=True).click()
                    assert page.evaluate("window.originalVideo === document.querySelector('video') && !document.querySelector('video').paused")
                    page.get_by_role('button', name='Pause', exact=True).click()
                    assert marker.get_attribute('cx') == selected
                    page.get_by_label('Time (s)').fill('1.2')
                    page.wait_for_function("!document.querySelector('video').seeking")
                    page.get_by_role('button', name='Dual lens', exact=True).click()
                    page.get_by_role('button', name='RIGHT', exact=True).click()
                    page.locator('canvas[aria-label="Dual Lens view"]').click(position={'x': 400, 'y': 250})
                    page.get_by_role('button', name='Measurements · 2 points').wait_for()
                    page.get_by_role('button', name='Seguimiento de balón', exact=True).click()
                    page.wait_for_function("Math.abs(document.querySelector('video').currentTime-1.2)<0.02 && !document.querySelector('video').seeking && document.querySelector('video').readyState>=2")
                    with page.expect_response('**/tracking-point') as response:
                        page.locator('video').click()
                    tracking = response.value.json()
                    assert tracking['timestamp_s'] == 1.2 and tracking['row_index'] == 35
                    page.get_by_role('button', name='Measurements · 3 points').wait_for()
                    assert marker.count() == 1
                    assert abs(float(marker.get_attribute('cx')) - tracking['world_x']) < 1e-6
                    # Force letterboxing, then prove a bar click sends no request
                    # and keeps the current selected pitch point.
                    tracking_calls = []
                    page.on('request', lambda req: tracking_calls.append(req) if req.url.endswith('/tracking-point') else None)
                    page.locator('video').evaluate("v => {v.style.height='300px';v.style.maxHeight='none'}")
                    page.locator('video').click(position={'x': 5, 'y': 150})
                    page.wait_for_timeout(100)
                    assert not tracking_calls
                    page.get_by_label('Time (s)').fill('2.5')
                    page.wait_for_function("!document.querySelector('video').seeking")
                    with page.expect_response('**/tracking-point') as response:
                        page.locator('video').click()
                    assert response.value.json()['status'] == 'camera_state_unavailable'
                    assert page.get_by_role('button', name='Measurements · 3 points').count() == 1
                    assert marker.count() == 1
                    page.get_by_label('Time (s)').fill('1.2')
                    page.wait_for_function("!document.querySelector('video').seeking")
                    page.get_by_role('button', name='Panorama', exact=True).click()
                    page.wait_for_function("document.querySelector('video')?.readyState >= 2 && !document.querySelector('video').seeking")
                    page.locator('canvas[aria-label="Panorama view"]').click(position={'x': 500, 'y': 320})
                    page.get_by_role('button', name='Measurements · 4 points').wait_for()
                    page.get_by_role('button', name='Play', exact=True).click()
                    page.wait_for_function("document.querySelector('video').currentTime > 1.5")
                    page.get_by_role('button', name='Pause', exact=True).click()
                    if captures:
                        page.screenshot(path=str(captures / 'workspace.png'))
                        page.get_by_role('button', name='Manage files', exact=True).click()
                        page.get_by_role('dialog').wait_for()
                        page.screenshot(path=str(captures / 'manage-files.png'), animations='disabled')
                        page.get_by_role('button', name='Done', exact=True).click()
                    page.set_viewport_size({'width': 900, 'height': 1000})
                    page.get_by_role('button', name='Interactive', exact=True).click()
                    page.wait_for_function("Math.abs(document.querySelector('canvas[aria-label=\"Interactive view\"]').width-document.querySelector('canvas[aria-label=\"Interactive view\"]').getBoundingClientRect().width)<2")
                    assert interactive.bounding_box()['width'] < 900
                    assert page.evaluate('document.documentElement.scrollWidth <= innerWidth')
                    # Pick after resizing with the CURRENT GPU camera (after drag
                    # and zoom), independently reconstructing the ray from uniforms.
                    picks = page.evaluate('''context => {
                      const c=document.querySelector('canvas[aria-label="Interactive view"]');
                      const gl=c.getContext('webgl'), p=gl.getParameter(gl.CURRENT_PROGRAM);
                      const m=gl.getUniform(p,gl.getUniformLocation(p,'interactive_to_world'));
                      const scale=gl.getUniform(p,gl.getUniformLocation(p,'interactive_scale'));
                      const origin=context.uniforms.camera2world.value.slice(12,15), b=c.getBoundingClientRect();
                      let valid=null, invalid=null;
                      for(let y=1;y<c.height;y+=20) for(let x=1;x<c.width;x+=20) {
                        const v=[(2*(x+.5)/c.width-1)*scale[0],(2*(y+.5)/c.height-1)*scale[1],1];
                        const d=[0,1,2].map(row=>v.reduce((s,a,k)=>s+a*m[k*4+row],0));
                        const t=-origin[1]/d[1], wx=origin[0]+t*d[0], wz=origin[2]+t*d[2];
                        const click={x:(x+.5)/c.width*b.width,y:(y+.5)/c.height*b.height,wx,wz};
                        if(t>0 && Math.abs(wx)<context.field_length/2-1 && Math.abs(wz)<context.field_width/2-1) valid??=click;
                        else if(t<=0 || Math.abs(wx)>context.field_length/2+1 || Math.abs(wz)>context.field_width/2+1) invalid??=click;
                      }
                      return {valid,invalid};
                    }''', view_context(str(calibrations[0]['path'])))
                    assert picks['valid'] and picks['invalid']
                    interactive.click(position={k: picks['valid'][k] for k in ('x', 'y')})
                    page.get_by_role('button', name='Measurements · 5 points').wait_for()
                    assert marker.count() == 1
                    assert abs(float(marker.get_attribute('cx')) - picks['valid']['wx']) < 1e-4
                    assert abs(float(marker.get_attribute('cy')) - picks['valid']['wz']) < 1e-4
                    selected = marker.get_attribute('cx')
                    interactive.click(position={k: picks['invalid'][k] for k in ('x', 'y')})
                    assert page.get_by_role('button', name='Measurements · 5 points').count() == 1
                    assert marker.get_attribute('cx') == selected
                    if captures:
                        page.screenshot(path=str(captures / 'interactive.png'), animations='disabled')
                    assert page.locator('svg[aria-label="Pitch reference"]').count() == 1
                    assert not errors, errors
                    print('Browser workflow: four views, shared clock/marker, drag vs click, zoom, playback, Tracking .det, black bars, invalid samples and resize passed', flush=True)
                    browser.close()
            except Exception:
                print((work / 'servers.log').read_text(), file=sys.stderr)
                raise
            finally:
                frontend.terminate(); backend.terminate()
                frontend.wait(timeout=10); backend.wait(timeout=10)


if __name__ == '__main__':
    main()
