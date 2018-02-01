#version 300 es


precision highp float;

uniform float u_Time;
uniform vec2 u_AspectRatio;
uniform vec4 u_Eye;

out vec4 out_Col;
in vec2 f_Pos;
in float aspectRatio;

// referenced this tutorial: http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/

const int MAX_MARCHING_STEPS = 255;
const float MIN_DIST = 0.0;
const float MAX_DIST = 100.0;
const float EPSILON = 0.0001;


//const vec4 cameraPos = vec4()
//const vec4 lightPos = vec4()

float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

float sceneSDF(vec3 p)
{
	return sphereSDF(p);
}


/**
https://www.shadertoy.com/view/llt3R4
 * Return the shortest distance from the eyepoint to the scene surface along
 * the marching direction. If no part of the surface is found between start and end,
 * return end.
 * 
 * eye: the eye point, acting as the origin of the ray
 * marchingDirection: the normalized direction to march in
 * start: the starting distance away from the eye
 * end: the max distance away from the ey to march before giving up
 */
float shortestDistanceToSurface(vec3 eye, vec3 marchingDirection, float start, float end) {
    float depth = start;
    for (int i = 0; i < MAX_MARCHING_STEPS; i++) {
        float dist = sceneSDF(eye + depth * marchingDirection);
        if (dist < EPSILON) {
			return depth;
        }
        depth += dist;
        if (depth >= end) {
            return end;
        }
    }
    return end;
}
   

/**
https://www.shadertoy.com/view/llt3R4
 * Return the normalized direction to march in from the eye point for a single pixel.
 * 
 * fieldOfView: vertical field of view in degrees
 * size: resolution of the output image
 * fragCoord: the x,y coordinate of the pixel in the output image
 */
vec3 rayDirection(float fieldOfView, vec2 size, vec2 fragCoord) {
    vec2 xy = fragCoord - size / 2.0;
    float z = size.y / tan(radians(fieldOfView) / 2.0);
    return normalize(vec3(xy, -z));
}

//https://www.shadertoy.com/view/llt3R4
void main() {
	// TODO: make a Raymarcher!

	vec2 fragCoord = ((f_Pos.xy + 1.0) / 2.0) * u_AspectRatio.xy;
	vec3 dir = rayDirection(45.0, u_AspectRatio, fragCoord);
    vec3 eye = vec3(0.0, 0.0, 5.0); // Doesn't work if I say eye = vec3(u_Eye);, figure out why
    float dist = shortestDistanceToSurface(eye, dir, MIN_DIST, MAX_DIST);
    
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        out_Col = vec4(0.0, 0.0, 0.0, 1.0);
		return;
    }
    
    out_Col = vec4(1.0, 0.0, 0.0, 1.0);
	
	//out_Col = vec4(vec3(u_Time / 1000.0), 1.0);
	//out_Col = vec4(vec3(length(aspectRatio * f_Pos)),1.0);//vec4(1.0, 0.5, 0.0, 1.0);
}