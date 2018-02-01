#version 300 es

#define PI 3.14159265359
#define deg2rad PI / 180.0
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

float time;


//const vec4 cameraPos = vec4()
//const vec4 lightPos = vec4()

vec2 convertAspectRatio(vec2 st) {

	return ((st + 1.0) / 2.0) * u_AspectRatio.xy;
   // return 2.0 * (st - 0.5) * u_AspectRatio.xy / max(u_AspectRatio.x, u_AspectRatio.y);
}


//http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
float intersectSDF(float distA, float distB) {
    return max(distA, distB);
}

//http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
float unionSDF(float distA, float distB) {
    return min(distA, distB);
}

//http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
float differenceSDF(float distA, float distB) {
    return max(distA, -distB);
}

//http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
float sphereSDF(vec3 p) {
    return length(p) - 1.0;
}

/**
 * Rotation matrix around the X axis. https://www.shadertoy.com/view/4tcGDr
 */
mat3 rotateX(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(1, 0, 0),
        vec3(0, c, -s),
        vec3(0, s, c)
    );
}

/**
 * Rotation matrix around the Y axis. https://www.shadertoy.com/view/4tcGDr
 */
mat3 rotateY(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, 0, s),
        vec3(0, 1, 0),
        vec3(-s, 0, c)
    );
}

/**
 * Rotation matrix around the Z axis. https://www.shadertoy.com/view/4tcGDr
 */
mat3 rotateZ(float theta) {
    float c = cos(theta);
    float s = sin(theta);
    return mat3(
        vec3(c, -s, 0),
        vec3(s, c, 0),
        vec3(0, 0, 1)
    );
}

/**
 * Signed distance function for a sphere centered at the origin with radius r.
 */
float sphereSDF(vec3 p, float r) {
    return length(p) - r;
}

// iq
float capsuleSDF( vec3 p, vec3 a, vec3 b, float r )
{
    vec3 pa = p - a;
	vec3 ba = b - a;
    float h = clamp( dot(pa,ba)/dot(ba,ba), 0.0, 1.0 );
    return length( pa - ba*h ) - r;
}

/**
 * Signed distance function for an XY aligned cylinder centered at the origin with
 * height h and radius r. https://www.shadertoy.com/view/4tcGDr 
 */
float cylinderSDF(vec3 p, float h, float r) {
    // How far inside or outside the cylinder the point is, radially
    float inOutRadius = length(p.xy) - r;
    
    // How far inside or outside the cylinder is, axially aligned with the cylinder
    float inOutHeight = abs(p.z) - h/2.0;
    
    // Assuming p is inside the cylinder, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(inOutRadius, inOutHeight), 0.0);

    // Assuming p is outside the cylinder, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(vec2(inOutRadius, inOutHeight), 0.0));
    
    return insideDistance + outsideDistance;
}

/**https://www.shadertoy.com/view/Xtd3z7
 * Signed distance function for a cube centered at the origin
 * with width = height = length = 2.0
 */
/**
 * Signed distance function for a cube centered at the origin
 * with dimensions specified by size.
 */
float cubeSDF(vec3 p, vec3 size) {
    vec3 d = abs(p) - (size / 2.0);
    
    // Assuming p is inside the cube, how far is it from the surface?
    // Result will be negative or zero.
    float insideDistance = min(max(d.x, max(d.y, d.z)), 0.0);
    
    // Assuming p is outside the cube, how far is it from the surface?
    // Result will be positive or zero.
    float outsideDistance = length(max(d, 0.0));
    
    return insideDistance + outsideDistance;
}


float testSceneSDF(vec3 samplePoint)
{
	// Slowly spin the whole scene
    samplePoint = rotateY(time / 2.0) * samplePoint;
    
    float cylinderRadius = 0.4 + (1.0 - 0.4) * (1.0 + sin(1.7 * time)) / 2.0;
    float cylinder1 = cylinderSDF(samplePoint, 2.0, cylinderRadius);
    float cylinder2 = cylinderSDF(rotateX(radians(90.0)) * samplePoint, 2.0, cylinderRadius);
    float cylinder3 = cylinderSDF(rotateY(radians(90.0)) * samplePoint, 2.0, cylinderRadius);
    
    float cube = cubeSDF(samplePoint, vec3(1.8, 1.8, 1.8));
    
    float sphere = sphereSDF(samplePoint, 1.2);
    
    float ballOffset = 0.4 + 1.0 + sin(1.7 * time);
    float ballRadius = 0.3;
    float balls = sphereSDF(samplePoint - vec3(ballOffset, 0.0, 0.0), ballRadius);
    balls = unionSDF(balls, sphereSDF(samplePoint + vec3(ballOffset, 0.0, 0.0), ballRadius));
    balls = unionSDF(balls, sphereSDF(samplePoint - vec3(0.0, ballOffset, 0.0), ballRadius));
    balls = unionSDF(balls, sphereSDF(samplePoint + vec3(0.0, ballOffset, 0.0), ballRadius));
    balls = unionSDF(balls, sphereSDF(samplePoint - vec3(0.0, 0.0, ballOffset), ballRadius));
    balls = unionSDF(balls, sphereSDF(samplePoint + vec3(0.0, 0.0, ballOffset), ballRadius));
    
    
    
    float csgNut = differenceSDF(intersectSDF(cube, sphere),
                         unionSDF(cylinder1, unionSDF(cylinder2, cylinder3)));
    
    return unionSDF(balls, csgNut);
}

vec3 scaleOp(vec3 samplePoint, vec3 scale)
{
	return (samplePoint / scale) * min(scale.x, min(scale.y, scale.z));
}

// iq
float smin( float a, float b, float k )
{
    float h = clamp( 0.5+0.5*(b-a)/k, 0.0, 1.0 );
    return mix( b, a, h ) - k*h*(1.0-h);
}

float skull(vec3 p)
{
	vec3 scale = vec3(2.0, 1.5, 1.0); // scale sphere to ellipsoid
	vec3 headPoint = scaleOp(p, scale);
	headPoint -= vec3(0.0, .03, 0.0); //translate up
	float circle = sphereSDF(headPoint, .10);

	headPoint -= vec3(0.0, -.06, 0.0); 
	headPoint = scaleOp(headPoint, vec3(.5, .5, .6));
	float square = cubeSDF(headPoint, vec3(.1, .1, .1)) - .01; // subtract a small constant for rounded edges

	return smin(circle, square, .01);
}

float eyes(vec3 p)
{
	//eyeball
	vec3 scale = vec3(1.7, 2.5, 1.0); // scale
	vec3 eye = scaleOp(p, scale);
	vec3 eye1 = eye - vec3(0.03, .0, 0.09); //translate
	vec3 eye2 = eye - vec3(-0.03, .0, 0.09);
	float e1 = sphereSDF(eye1, .03);
	float e2 = sphereSDF(eye2, .03);
	float eyes = unionSDF(e1, e2);

	// pupils
	vec3 pupil1 = eye1;//scaleOp(eye1, vec3(1.0, 1.0, 1.0));
	vec3 pupil2 = eye2;//scaleOp(eye2, vec3(1.0, 1.0, 1.0));
	pupil1 -= vec3(0.0, -.01, 0.0);
	pupil2 -= vec3(0.0, -.01, 0.0);
	float p1 = cubeSDF(pupil1, vec3(.01, .02, 0.08));
	float p2 = cubeSDF(pupil2, vec3(.01, .02, .08));
	float pupils = unionSDF(p1, p2);

	// eyelid molded from a block
	//vec3 eyelid = scaleOp()
	// vec3 lid1 = scaleOp(eye1 - vec3(0.0, 0.01, .015), vec3(1.0, 1.0, 1.0));
	// lid1 -= vec3(0.0, .01, 0.0);
	// lid1 = rotateX(-15.0 * deg2rad) * lid1;
	// float l1 = cubeSDF(lid1, vec3(.05, .035, .025));

	// float eyeAndLid = intersectSDF(eyes, l1);
	float eyeballs = differenceSDF(eyes, pupils);
	return eyeballs;//min(eyeAndLid, eyeballs);//unionSDF(eyeAndLid, eyeballs);
}

float mouth(vec3 p)
{
	vec3 translate = vec3(0, -.65, 0.0);
	//samp = scaleOp(samp, vec3(1.5, 1.5, .5));
//	samp = rotateX(90.0 * deg2rad) * samp;
//	samp -= vec3(0, -0.13, 0.0);
//	samp = scaleOp(samp, vec3(.01, .01, .01));

	vec3 a = vec3(-.15, 0.5, 0.0);
	vec3 b = vec3(.15, 0.5, 0.0);;
	float r = .06;
	
	float mouth = capsuleSDF(p - translate, a, b, r);//cylinderSDF(samp, .3, .031);//capsuleSDF(p, a, b, r);
	return mouth;
}

float head(vec3 p)
{
	
	float skull = skull(p);
	float eyes = eyes(p);
	float mouth = mouth(p);
	float skullAndEyes = unionSDF(skull, eyes);
	return smin(skullAndEyes, mouth, .02);
}

float squidward(vec3 samplePoint)
{
	// Slowly spin the whole scene
    //samplePoint = rotateY(time / 2.0) * samplePoint;

	// HEAD
	// sphere sdf scaled, translated, rotated
	
	return head(samplePoint);

	// EYES
	// sphere sdf, scaled, translated, rotated + cube sdfs scaled, translated, union with sphere
	// NOSE
	// help on this... bend a cylinder and intersect with a sphere? 
	

}

float sceneSDF(vec3 samplePoint)
{
	return squidward(samplePoint);
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

/**
http://jamie-wong.com/2016/07/15/ray-marching-signed-distance-functions/
 * Using the gradient of the SDF, estimate the normal on the surface at point p.
 */
vec3 estimateNormal(vec3 p) {
    return normalize(vec3(
        sceneSDF(vec3(p.x + EPSILON, p.y, p.z)) - sceneSDF(vec3(p.x - EPSILON, p.y, p.z)),
        sceneSDF(vec3(p.x, p.y + EPSILON, p.z)) - sceneSDF(vec3(p.x, p.y - EPSILON, p.z)),
        sceneSDF(vec3(p.x, p.y, p.z  + EPSILON)) - sceneSDF(vec3(p.x, p.y, p.z - EPSILON))
    ));
}


/**
https://www.shadertoy.com/view/lt33z7
 * Lighting contribution of a single point light source via Phong illumination.
 * 
 * The vec3 returned is the RGB color of the light's contribution.
 *
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 * lightPos: the position of the light
 * lightIntensity: color/intensity of the light
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongContribForLight(vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye,
                          vec3 lightPos, vec3 lightIntensity) {
    vec3 N = estimateNormal(p);
    vec3 L = normalize(lightPos - p);
    vec3 V = normalize(eye - p);
    vec3 R = normalize(reflect(-L, N));
    
    float dotLN = dot(L, N);
    float dotRV = dot(R, V);
    
    if (dotLN < 0.0) {
        // Light not visible from this point on the surface
        return vec3(0.0, 0.0, 0.0);
    } 
    
    if (dotRV < 0.0) {
        // Light reflection in opposite direction as viewer, apply only diffuse
        // component
        return lightIntensity * (k_d * dotLN);
    }
    return lightIntensity * (k_d * dotLN + k_s * pow(dotRV, alpha));
}

/**
https://www.shadertoy.com/view/lt33z7
 * Lighting via Phong illumination.
 * 
 * The vec3 returned is the RGB color of that point after lighting is applied.
 * k_a: Ambient color
 * k_d: Diffuse color
 * k_s: Specular color
 * alpha: Shininess coefficient
 * p: position of point being lit
 * eye: the position of the camera
 *
 * See https://en.wikipedia.org/wiki/Phong_reflection_model#Description
 */
vec3 phongIllumination(vec3 k_a, vec3 k_d, vec3 k_s, float alpha, vec3 p, vec3 eye) {
    const vec3 ambientLight = 0.5 * vec3(1.0, 1.0, 1.0);
    vec3 color = ambientLight * k_a;
    
    vec3 light1Pos = vec3(4.0 * sin(time),
                          2.0,
                          4.0 * cos(time));    vec3 light1Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light1Pos,
                                  light1Intensity);
    
    vec3 light2Pos = vec3(2.0 * sin(0.37 * time),
                         2.0 * cos(0.37 * time),
                         2.0);
    vec3 light2Intensity = vec3(0.4, 0.4, 0.4);
    
    color += phongContribForLight(k_d, k_s, alpha, p, eye,
                                  light2Pos,
                                  light2Intensity);    
    return color;
}

/** https://www.shadertoy.com/view/Xtd3z7
 * Return a transform matrix that will transform a ray from view space
 * to world coordinates, given the eye point, the camera target, and an up vector.
 *
 * This assumes that the center of the camera is aligned with the negative z axis in
 * view space when calculating the ray marching direction. See rayDirection.
 */
mat4 viewMatrix(vec3 eye, vec3 center, vec3 up) {
    // Based on gluLookAt man page
    vec3 f = normalize(center - eye);
    vec3 s = normalize(cross(f, up));
    vec3 u = cross(s, f);
    return mat4(
        vec4(s, 0.0),
        vec4(u, 0.0),
        vec4(-f, 0.0),
        vec4(0.0, 0.0, 0.0, 1)
    );
}

//https://www.shadertoy.com/view/llt3R4
void main() {
	// TODO: make a Raymarcher!

// control animation angle
	time = 2.0;//u_Time / 100.0;

	vec2 fragCoord = convertAspectRatio(f_Pos.xy);
	vec3 viewDir = rayDirection(45.0, u_AspectRatio, fragCoord);
   //vec3 eye = vec3(0.0, 0.0, 5.0); // Doesn't work if I say eye = vec3(u_Eye);, figure out why
    //vec3 eye = vec3(0.0, 0.0, 1.0);
	vec3 eye = vec3(0.0, 0.0, 5.0);
    
    mat4 viewToWorld = viewMatrix(eye, vec3(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0));
    
    vec3 worldDir = (viewToWorld * vec4(viewDir, 0.0)).xyz;
    
    float dist = shortestDistanceToSurface(eye, worldDir, MIN_DIST, MAX_DIST);
    
  //  float dist = shortestDistanceToSurface(eye, dir, MIN_DIST, MAX_DIST);
    
    if (dist > MAX_DIST - EPSILON) {
        // Didn't hit anything
        out_Col = vec4(0.0, 0.0, 0.0, 1.0);
		return;
    }
    
    // The closest point on the surface to the eyepoint along the view ray
    vec3 p = eye + dist * worldDir;
    
    vec3 K_a = vec3(0.2, 0.2, 0.2);
    vec3 K_d = vec3(0.7, 0.2, 0.2);
    vec3 K_s = vec3(1.0, 1.0, 1.0);
    float shininess = 10.0;
    
    vec3 color = phongIllumination(K_a, K_d, K_s, shininess, p, eye);
    
    out_Col = vec4(color, 1.0);
	
	//out_Col = vec4(vec3(u_Time / 1000.0), 1.0);
	//out_Col = vec4(vec3(length(aspectRatio * f_Pos)),1.0);//vec4(1.0, 0.5, 0.0, 1.0);
}