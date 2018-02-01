#version 300 es

precision highp float;

in vec4 vs_Pos;

uniform float u_Time;
uniform vec2 u_AspectRatio;
uniform vec4 u_Eye;

out vec2 f_Pos;
out float aspectRatio;

void main() {
	// TODO: Pass relevant info to fragment
	gl_Position = vs_Pos;
	f_Pos = vs_Pos.xy; 
	aspectRatio = u_AspectRatio.x / u_AspectRatio.y;
	
}
