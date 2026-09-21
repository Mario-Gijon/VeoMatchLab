#define M_PI 3.14159265358979323846
#define aspect 16.0/9.0
precision highp float;
#define GLSLIFY 1

varying vec2 v_texcoord;
uniform sampler2D texture;

uniform mat4 camera2world;
uniform mat4 camera2logic;
uniform mat4 camera2presentation;

uniform mat4 lens_left2camera;
uniform mat4 lens_right2camera;

//Field dimensions
uniform float field_length;
uniform float field_width;

uniform float va_left;
uniform float va_right;
uniform float va_bottom;
uniform float va_top;

// Intrinsics per lense per camera
uniform mat3 intrinsic_left;
uniform mat3 intrinsic_right;

struct DistortionParameters {
  float k1;
  float k2;
  float p1;
  float p2;
  float k3;
  float k4;
  float k5;
  float k6;
};

uniform DistortionParameters distortion_left;
uniform DistortionParameters distortion_right;

uniform sampler2D watermarkTexture;

vec4 getImage(sampler2D image, vec2 uv, bool isLeft){
  float topDestRatio = 19.0 / 45.0;
  float topSourceRatio = 19.0 / 32.0;
  float bottomDestRatio = 1.0 - topDestRatio;
  float bottomSourceRatio = 1.0 - topSourceRatio;


  if (uv.y < topDestRatio){
    uv.y = uv.y / topDestRatio * (topSourceRatio);
  } else {
    uv.y = (uv.y - topDestRatio) / bottomDestRatio * bottomSourceRatio + topSourceRatio;
  }

  // to not renderd if it is outside of view
  if( uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 ) {
    return vec4(0.0, 0.0, 0.0, 0.0);
  }

  uv.y = uv.y * 0.5;

  if (!isLeft){
    uv.y = uv.y + 0.5;
  }

  return texture2D(image, uv);
}

mat4 inverse(mat4 m) {
  float
      a00 = m[0][0], a01 = m[0][1], a02 = m[0][2], a03 = m[0][3],
      a10 = m[1][0], a11 = m[1][1], a12 = m[1][2], a13 = m[1][3],
      a20 = m[2][0], a21 = m[2][1], a22 = m[2][2], a23 = m[2][3],
      a30 = m[3][0], a31 = m[3][1], a32 = m[3][2], a33 = m[3][3],

      b00 = a00 * a11 - a01 * a10,
      b01 = a00 * a12 - a02 * a10,
      b02 = a00 * a13 - a03 * a10,
      b03 = a01 * a12 - a02 * a11,
      b04 = a01 * a13 - a03 * a11,
      b05 = a02 * a13 - a03 * a12,
      b06 = a20 * a31 - a21 * a30,
      b07 = a20 * a32 - a22 * a30,
      b08 = a20 * a33 - a23 * a30,
      b09 = a21 * a32 - a22 * a31,
      b10 = a21 * a33 - a23 * a31,
      b11 = a22 * a33 - a23 * a32,

      det = b00 * b11 - b01 * b10 + b02 * b09 + b03 * b08 - b04 * b07 + b05 * b06;

  return mat4(
      a11 * b11 - a12 * b10 + a13 * b09,
      a02 * b10 - a01 * b11 - a03 * b09,
      a31 * b05 - a32 * b04 + a33 * b03,
      a22 * b04 - a21 * b05 - a23 * b03,
      a12 * b08 - a10 * b11 - a13 * b07,
      a00 * b11 - a02 * b08 + a03 * b07,
      a32 * b02 - a30 * b05 - a33 * b01,
      a20 * b05 - a22 * b02 + a23 * b01,
      a10 * b10 - a11 * b08 + a13 * b06,
      a01 * b08 - a00 * b10 - a03 * b06,
      a30 * b04 - a31 * b02 + a33 * b00,
      a21 * b02 - a20 * b04 - a23 * b00,
      a11 * b07 - a10 * b09 - a12 * b06,
      a00 * b09 - a01 * b07 + a02 * b06,
      a31 * b01 - a30 * b03 - a32 * b00,
      a20 * b03 - a21 * b01 + a22 * b00) / det;
}

vec3 distort(vec3 i, mat3 intrinsic, DistortionParameters distortion){
  // Normalize coordinates
  float x = i.x;
  float y = i.y;

  // Square distance powers
  float r2 = x*x + y*y;
  float r4 = r2 * r2;
  float r6 = r4 * r2;

  // Radial distortion
  float radial_xy = (1.0 + distortion.k1 * r2 + distortion.k2 * r4 + distortion.k3 * r6) / (1.0 + distortion.k4 * r2  + distortion.k5 * r4 + distortion.k6 * r6 );

  // Tangential
  float tangential_x = (distortion.p2 * (r2 + 2.0 * x * x) + 2.0 * distortion.p1 * x * y);
  float tangential_y = (distortion.p1 * (r2 + 2.0 * y * y) + 2.0 * distortion.p2 * x * y);

  return intrinsic * vec3(x * radial_xy + tangential_x, y * radial_xy + tangential_y, 1.0);
}

vec2 getUV(vec4 camspace_dir, mat3 intrinsic, DistortionParameters distortion){
  camspace_dir /= camspace_dir.z;
  vec3 ideal_pixel = vec3(camspace_dir.xy, 1.0);
  vec3 coord = distort(ideal_pixel, intrinsic, distortion);
  //Intrinsic matrix is built for 4K image
  return coord.xy / vec2(1920.0 * 2.0, 1080.0 * 2.0);
}

/** END **/

mat4 rotMatX(float rads){
  float sss = sin(rads);float ccc = cos(rads);
  return mat4(1.0, 0.0, 0.0, 0.0,
              0.0, ccc, sss, 0.0,
              0.0,-sss, ccc, 0.0,
              0.0, 0.0, 0.0, 1.0
              );
}

mat4 rotMatY(float rads){
  float sss = sin(rads);float ccc = cos(rads);
  return mat4(ccc, 0.0,-sss, 0.0,
              0.0, 1.0, 0.0, 0.0,
              sss, 0.0, ccc, 0.0,
              0.0, 0.0, 0.0, 1.0);
}

mat4 rotMatZ(float rads){
  float sss = sin(rads);float ccc = cos(rads);
  return mat4(ccc, sss, 0.0, 0.0,
             -sss, ccc, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0);
}

void main(){

  gl_FragColor = vec4(0.0, 0.0, 0.0, 1.0);

  vec2 uv = v_texcoord;

  vec4 worldspace_projection_dir;
  vec4 worldspace_projection_dir_right;
  vec4 worldspace_projection_dir_left;

  vec4 camspace_dir_right;
  vec4 camspace_dir_left;

  //The position in the scene we are analyzing
  vec4 projection_target_point;
  vec4 blend_target_point;

  float min_span = 0.005;
  float blend_distance = 0.3;      // meters
  float projection_bubble_radius = 1000.0; //Meters
  float blend_bubble_radius = 20.0; //Meters

  //Position of projection camera
  vec4 worldspace_cam_pos = camera2world * vec4(0.0 ,0.0 ,0.0, 1.0);
  vec4 worldspace_lens_pos_left = camera2world * lens_left2camera * vec4(0.0 ,0.0 ,0.0, 1.0);
  vec4 worldspace_lens_pos_right = camera2world * lens_right2camera * vec4(0.0 ,0.0 ,0.0, 1.0);

  // *** Calculation for PANORAMA ONLY *** //
  //forward direction
  //Find uv coordinates in region of interest
  float lon = uv.x * (va_right - va_left) + va_left;
  float lat = uv.y * (va_top - va_bottom) + va_bottom;

  worldspace_projection_dir = camera2world * camera2presentation * rotMatY(lon) * rotMatX(lat) * vec4(0.0, 0.0, 1.0, 0.0);
  // *** END *** //

    //Find worldpoint to project to
  float up_down = (sign(worldspace_projection_dir.y) + 1.0) / 2.0;

  vec4 projection_direction = worldspace_projection_dir * (up_down * (-worldspace_cam_pos.y/worldspace_projection_dir.y) + (1.0-up_down)  * projection_bubble_radius);
  projection_target_point = worldspace_cam_pos + projection_direction;
  projection_target_point.w = 1.0;

  blend_target_point = inverse(camera2world * camera2presentation) * projection_direction;
  blend_target_point.w = 1.0;

  if (length(projection_direction) > blend_bubble_radius || projection_direction.y < 0.0){
    blend_target_point = blend_target_point / length(blend_target_point) * blend_bubble_radius;
  }

  float blend = clamp(blend_target_point.x/blend_distance + 0.5, 0.0, 1.0);

  //Left
  worldspace_projection_dir_left = projection_target_point - worldspace_lens_pos_left;
  camspace_dir_left =  inverse(camera2world * lens_left2camera) * worldspace_projection_dir_left;

  vec2 left_uv = getUV(camspace_dir_left, intrinsic_left, distortion_left);
  vec4 left_color = getImage(texture, left_uv, true);

  //Right
  worldspace_projection_dir_right = projection_target_point - worldspace_lens_pos_right;
  camspace_dir_right =  inverse(camera2world * lens_right2camera) * worldspace_projection_dir_right;

  vec2 right_uv = getUV(camspace_dir_right, intrinsic_right, distortion_right);
  vec4 right_color = getImage(texture, right_uv, false);

  // Do the blend
  vec4 magic_c = left_color * (1.0 - blend) + right_color * blend;
  //------------------------------
  // OVERLAYS
  vec4 watermark = texture2D(watermarkTexture, vec2(v_texcoord.x, 1.0 - v_texcoord.y));

  magic_c.xyz = mix(vec3(0.0, 0.0, 0.0), magic_c.xyz, 1.0);
  gl_FragColor = magic_c * (1.0 - watermark.a) + watermark * watermark.a;

  gl_FragColor.a = 1.0;
}
