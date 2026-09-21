import type { ViewContext } from '../types'
import { interactiveShader, INTERACTIVE_UNIFORMS } from './interactiveCamera'

/** Feed the captured shaders unchanged. Python supplies column-major uniforms,
 * including the validated inverse camera2presentation convention. */
export function createPanoramaGL(canvas: HTMLCanvasElement, context: ViewContext, enableInteractive = false) {
  const gl = canvas.getContext('webgl', { alpha: false, preserveDrawingBuffer: true })
  if (!gl) throw new Error('WebGL no está disponible. Usa el renderizado de fotogramas.')
  const shaders: WebGLShader[] = []
  const buffers: WebGLBuffer[] = []
  const textures: WebGLTexture[] = []
  const program = gl.createProgram()!
  const dispose = () => {
    buffers.forEach((buffer) => gl.deleteBuffer(buffer))
    textures.forEach((texture) => gl.deleteTexture(texture))
    shaders.forEach((shader) => gl.deleteShader(shader))
    gl.deleteProgram(program)
  }
  try {
    const compile = (type: number, source: string) => {
      const shader = gl.createShader(type)!
      shaders.push(shader)
      gl.shaderSource(shader, source); gl.compileShader(shader)
      if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(shader) || 'Shader compilation failed')
      return shader
    }
    const vertex = compile(gl.VERTEX_SHADER, context.vertex_shader)
    const fragmentSource = enableInteractive
      ? interactiveShader(context.fragment_shader).replace('void main(){', INTERACTIVE_UNIFORMS + '\nvoid main(){')
      : context.fragment_shader
    const fragment = compile(gl.FRAGMENT_SHADER, fragmentSource)
    gl.attachShader(program, vertex); gl.attachShader(program, fragment); gl.linkProgram(program)
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw new Error('Captured panorama shader could not link')
    gl.useProgram(program)
    const attribute = (name: string, values: number[]) => {
      const buffer = gl.createBuffer()!; buffers.push(buffer)
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer); gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(values), gl.STATIC_DRAW)
      const location = gl.getAttribLocation(program, name)
      gl.enableVertexAttribArray(location); gl.vertexAttribPointer(location, 2, gl.FLOAT, false, 0, 0)
    }
    attribute('a_position', [-1, -1, 1, -1, -1, 1, -1, 1, 1, -1, 1, 1])
    attribute('a_texcoord', [0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1])
    Object.entries(context.uniforms).forEach(([name, uniform]) => {
      const location = gl.getUniformLocation(program, name)
      if (uniform.matrix === 4) gl.uniformMatrix4fv(location, false, new Float32Array(uniform.value as number[]))
      else if (uniform.matrix === 3) gl.uniformMatrix3fv(location, false, new Float32Array(uniform.value as number[]))
      else gl.uniform1f(location, uniform.value as number)
    })
    const texture = (unit: number, name: string) => {
      const result = gl.createTexture()!
      textures.push(result)
      gl.activeTexture(gl.TEXTURE0 + unit); gl.bindTexture(gl.TEXTURE_2D, result)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE)
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE)
      gl.uniform1i(gl.getUniformLocation(program, name), unit)
      return result
    }
    const source = texture(0, 'texture')
    texture(1, 'watermarkTexture')
    gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, 1, 1, 0, gl.RGBA, gl.UNSIGNED_BYTE, new Uint8Array([0, 0, 0, 0]))
    const enabled = gl.getUniformLocation(program, 'interactive_enabled')
    const transform = gl.getUniformLocation(program, 'interactive_to_world')
    const scale = gl.getUniformLocation(program, 'interactive_scale')
    return {
      draw(image: TexImageSource, camera?: { matrix: number[]; fov: number }, upload = true) {
        gl.viewport(0, 0, canvas.width, canvas.height)
        gl.uniform1f(enabled, camera ? 1 : 0)
        if (camera) {
          gl.uniformMatrix4fv(transform, false, new Float32Array(camera.matrix))
          const tan = Math.tan(camera.fov / 2)
          gl.uniform2f(scale, tan * canvas.width / canvas.height, tan)
        }
        gl.activeTexture(gl.TEXTURE0); gl.bindTexture(gl.TEXTURE_2D, source)
        // Source texture y=0 is the TOP of the stacked video in the captured shader.
        gl.pixelStorei(gl.UNPACK_FLIP_Y_WEBGL, false)
        if (upload) gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, image)
        gl.drawArrays(gl.TRIANGLES, 0, 6)
      },
      dispose,
    }
  } catch (error) { dispose(); throw error }
}
