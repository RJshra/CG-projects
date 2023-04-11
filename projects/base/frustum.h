#pragma once

#include <iostream>
#include "plane.h"
#include "bounding_box.h"
//#include<math.h>
struct Frustum {
public:
	Plane planes[6];
	enum {
		LeftFace = 0,
		RightFace = 1,
		BottomFace = 2,
		TopFace = 3,
		NearFace = 4,
		FarFace = 5
	};
	float fabs(float a) const{
		return a > 0 ? a : -a;
	}
	bool intersect(const BoundingBox& aabb, const glm::mat4& modelMatrix) const {
		// TODO: judge whether the frustum intersects the bounding box
		// write your code here
		// ------------------------------------------------------------
		glm::vec3 center = (aabb.max + aabb.min) * 0.5f;
		glm::vec3 extents = aabb.max - center;
		glm::vec4 temp = modelMatrix * glm::vec4(center, 1.0f);
		center = glm::vec3(temp.x, temp.y, temp.z) / temp.w;
		glm::vec3 forward = planes[4].normal * extents.z;
		glm::vec3 right = glm::cross(planes[3].normal,planes[4].normal) * extents.x;
		glm::vec3 up = glm::cross(planes[4].normal, planes[1].normal) * extents.y;

		float newIi = fabs(glm::dot(glm::vec3(1.f, 0.f, 0.f), right)) +
			fabs(dot(glm::vec3(1.f, 0.f, 0.f), up)) +
			fabs(dot(glm::vec3(1.f, 0.f, 0.f), forward));

		float newIj = fabs(dot(glm::vec3(0.f, 1.f, 0.f), right)) +
			fabs(dot(glm::vec3(0.f, 1.f, 0.f), up)) +
			fabs(dot(glm::vec3(0.f, 1.f, 0.f), forward));

		float newIk = fabs(dot(glm::vec3(0.f, 0.f, 1.f), right)) +
			fabs(dot(glm::vec3(0.f, 0.f, 1.f), up)) +
			fabs(dot(glm::vec3(0.f, 0.f, 1.f), forward));

		extents = glm::vec3(newIi, newIj, newIk);
		for (int i = 0; i < 6; i++) {
			float r = extents.x * fabs(planes[i].normal.x) + extents.y * fabs(planes[i].normal.y)
				+ extents.z * fabs(planes[i].normal.z);
			if (-r > (dot(planes[i].normal, center) + planes[i].signedDistance))
				return 0;
		}
		return 1;
	
		// ------------------------------------------------------------
	}
};

inline std::ostream& operator<<(std::ostream& os, const Frustum& frustum) {
	os << "frustum: \n";
	os << "planes[Left]:   " << frustum.planes[0] << "\n";
	os << "planes[Right]:  " << frustum.planes[1] << "\n";
	os << "planes[Bottom]: " << frustum.planes[2] << "\n";
	os << "planes[Top]:    " << frustum.planes[3] << "\n";
	os << "planes[Near]:   " << frustum.planes[4] << "\n";
	os << "planes[Far]:    " << frustum.planes[5] << "\n";

	return os;
}