// TODO: Modify the following code to achieve GPU frustum culling
#version 330 core
layout(location = 0) in vec3 aPosition;
layout(location = 1) in mat4 aInstanceMatrix;
out int visible;
struct BoundingBox {
    vec3 min;
    vec3 max;
};

struct Plane {
    vec3 normal;
    float signedDistance;
};

struct Frustum {
    Plane planes[6];
};

uniform BoundingBox boundingBox;
uniform Frustum frustum;


float fabs(float a){
	return a>0?a:-a;
}

int intersect() {
		vec3 center = (boundingBox.max + boundingBox.min) * 0.5f;
		vec3 extents = boundingBox.max - center;
		vec4 temp = aInstanceMatrix * vec4(center, 1.0f);
		center = vec3(temp.x,temp.y,temp.z)/temp.w;
		vec3 forward = frustum.planes[4].normal * extents.z;
		vec3 right = cross(frustum.planes[3].normal,frustum.planes[4].normal) * extents.x;
		vec3 up = cross(frustum.planes[4].normal, frustum.planes[1].normal) * extents.y;
		
		float newIi = fabs(dot(vec3( 1.f, 0.f, 0.f ), right)) +
			fabs(dot(vec3( 1.f, 0.f, 0.f ), up)) +
			fabs(dot(vec3( 1.f, 0.f, 0.f ), forward));

		float newIj = fabs(dot(vec3( 0.f, 1.f, 0.f ), right)) +
			fabs(dot(vec3( 0.f, 1.f, 0.f ), up)) +
			fabs(dot(vec3( 0.f, 1.f, 0.f ), forward));

		float newIk = fabs(dot(vec3( 0.f, 0.f, 1.f ), right)) +
			fabs(dot(vec3( 0.f, 0.f, 1.f), up)) +
			fabs(dot(vec3( 0.f, 0.f, 1.f ), forward));

		extents = vec3( newIi,newIj,newIk );
		for (int i = 0; i < 6; i++) {
			float r = extents.x * fabs(frustum.planes[i].normal.x) + extents.y * fabs(frustum.planes[i].normal.y)
				+ extents.z * fabs(frustum.planes[i].normal.z);
			if (-r >( dot(frustum.planes[i].normal, center) + frustum.planes[i].signedDistance))
				return 0;
		}
		return 1;
	}

void main() {

    visible = intersect();

}