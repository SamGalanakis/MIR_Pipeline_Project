#version 330

in vec3 fragNormal;

out vec4 outColor;

uniform vec3 color;


void main()
{
    // vec3 ambientLightIntensity = vec3(0.3f, 0.2f, 0.4f) ;
    vec3 ambientLightIntensity = vec3(0.5f, 0.5f, 0.5f) ;
    vec3 sunLightIntensity = vec3(0.9f, 0.9f, 0.9f);
    vec3 sunLightDirection = normalize(vec3(-2.0f, -2.0f, 0.0f));
    
   

    vec3 lightIntensity = ambientLightIntensity + sunLightIntensity * max(dot(fragNormal, sunLightDirection), 0.0f);
    
    outColor = vec4(color * lightIntensity,1);
}