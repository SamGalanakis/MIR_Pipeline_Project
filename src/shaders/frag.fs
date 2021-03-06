#version 330

flat in vec3 fragNormal;

out vec4 outColor;

uniform vec3 color;


void main()
{

    vec3 ambientLightIntensity = vec3(0.6f, 0.6f, 0.6f) ;
    vec3 sunLightIntensity = vec3(1.0f, 1.0f, 1.0f);
    vec3 sunLightDirection = normalize(vec3(-2.0f, -0.0f, 0.0f));

    
    vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
    
    outColor = vec4(color * lightIntensity,1);
}