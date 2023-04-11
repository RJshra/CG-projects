#include <iostream>
#include "bvh.h"

void BVH::constructBVH(std::vector<Primitive>& primitives) {
    std::vector<PrimitiveInfo> primInfo;
    for (int i = 0; i < primitives.size(); ++i) {
        primInfo.push_back({ i, getAABB(primitives[i]) });
    }

    int totalNode = 0;
    BVHBuildNode* root = recursiveBuild(
        primitives, primInfo, 0, static_cast<int>(primInfo.size()), &totalNode);

    nodes.resize(totalNode);
    int offset = 0;
    toLinearTree(root, &offset);
}

glm::vec3 Offset(glm::vec3 p,AABB box) {
    glm::vec3 o = p - box.pMin;
    if (box.pMin.x < box.pMax.x)
        o.x /= (box.pMax.x - box.pMin.x);
    if (box.pMin.y < box.pMax.y)
        o.y /= (box.pMax.y - box.pMin.y);
    if (box.pMin.z < box.pMax.z)
        o.z /= (box.pMax.z - box.pMin.z);
    return o;
}

BVHBuildNode* BVH::recursiveBuild(
    const std::vector<Primitive>& primitives,
    std::vector<PrimitiveInfo>& primInfo,
    int start, int end, int* totalNodes
) {
    BVHBuildNode* node = new BVHBuildNode;
    *totalNodes += 1;
    int nPrimitives = end - start;
    if (nPrimitives == 1) {
        AABB box;
        int startId = static_cast<int>(orderedPrimitives.size());
        for (int i = start; i < end; ++i) {
            box = unionAABB(box, primInfo[i].box);
            orderedPrimitives.push_back(primitives[primInfo[i].pid]);
        }
        node->initLeafNode(box, startId, nPrimitives);
        return node;
    }
    else {
        // recursive build BVH 
        AABB centroidBounds, box;
        for (int i = start; i < end; ++i) {
            centroidBounds = unionAABB(centroidBounds, primInfo[i].centroid);
        }
        int dim = maximumDim(centroidBounds);
        int mid = (start + end) / 2;
        //if the longest edge is equal to the shortest edges, not split
        if (centroidBounds.pMax[dim] == centroidBounds.pMin[dim]) {
            int startId = static_cast<int>(orderedPrimitives.size());
            for (int i = start; i < end; i++) {
                box = unionAABB(box, primInfo[i].box);
                orderedPrimitives.push_back(primitives[primInfo[i].pid]);
            }
            node->initLeafNode(box, startId, nPrimitives);
            return node;
        }
        else {
            mid = (start + end) / 2;
            std::nth_element(&primInfo[start], &primInfo[mid],
                &primInfo[end - 1] + 1,
                [dim](const PrimitiveInfo& a,
                    const PrimitiveInfo& b) {
                        return a.centroid[dim] < b.centroid[dim];
                });
            node->initInteriorNode(recursiveBuild(primitives, primInfo, start, mid,
                                        totalNodes),
                                    recursiveBuild(primitives, primInfo, mid, end,
                                        totalNodes),dim);
        }

    } return node;
}
//        else {
//            //SAH
//            if (nPrimitives <= 2) {
//                mid = (start + end) / 2;
//                std::nth_element(&primInfo[start], &primInfo[mid],
//                    &primInfo[end - 1] + 1,
//                    [dim](PrimitiveInfo& a, PrimitiveInfo& b) {
//                        return a.centroid[dim] < b.centroid[dim];
//                    });
//
//            }
//            else {
//                //Allocate BucketInfo for SAH partition buckets
//                constexpr int nBuckets = 12;
//                struct BucketInfo {
//                    int count = 0;
//                    AABB bounds;
//                };
//                BucketInfo buckets[nBuckets];
//
//                //Initialize BucketInfo for SAH partition buckets
//                for (int i = start; i < end; ++i) {
//                    glm::vec3 ofs = primInfo[i].centroid - centroidBounds.pMin;
//                    if (centroidBounds.pMin.x < centroidBounds.pMax.x)
//                        ofs.x /= (centroidBounds.pMax.x - centroidBounds.pMin.x);
//                    if (centroidBounds.pMin.y < centroidBounds.pMax.y)
//                        ofs.y /= (centroidBounds.pMax.y - centroidBounds.pMin.y);
//                    if (centroidBounds.pMin.z < centroidBounds.pMax.z)
//                        ofs.z /= (centroidBounds.pMax.z - centroidBounds.pMin.z);
//                    int b = nBuckets * ofs[dim];
//                    if (b == nBuckets)
//                        b = nBuckets - 1;
//                    buckets[b].count++;
//                    buckets[b].bounds = unionAABB(buckets[b].bounds, primInfo[i].box);
//                }
//                //Compute costs for splitting after each bucket
//                float cost[nBuckets - 1];
//                for (int i = 0; i < nBuckets - 1; ++i) {
//                    AABB b0, b1;
//                    int count0 = 0, count1 = 0;
//                    for (int j = 0; j <= i; ++j) {
//                        b0 = unionAABB(b0, buckets[j].bounds);
//                        count0 += buckets[j].count;
//                    }
//                    for (int j = i + 1; j < nBuckets; ++j) {
//                        b1 = unionAABB(b1, buckets[j].bounds);
//                        count1 += buckets[j].count;
//                    }
//                    cost[i] = .125f + (count0 * b0.surfaceArea() +
//                        count1 * b1.surfaceArea()) / box.surfaceArea();
//                }
//
//                //Find bucket to split at that minimizes SAH metric
//                float minCost = cost[0];
//                int minCostSplitBucket = 0;
//                for (int i = 1; i < nBuckets - 1; ++i) {
//                    if (cost[i] < minCost) {
//                        minCost = cost[i];
//                        minCostSplitBucket = i;
//                    }
//                }
//                //Either create leaf or split primitives at selected SAH bucket
//                float leafCost = nPrimitives;
//                
//                if (nPrimitives > 19 || minCost < leafCost) {
//
//                    PrimitiveInfo* pmid = std::partition(&primInfo[start],
//                        &primInfo[end - 1] + 1,
//                        [=](PrimitiveInfo& pi) {
//                            int b = nBuckets * Offset(pi.centroid, centroidBounds)[dim];
//                            if (b == nBuckets) b = nBuckets - 1;
//                            return b <= minCostSplitBucket;
//                        });
//                    mid = pmid - &primInfo[0];
//                }
//                else {
//                    // << Create leaf BVHBuildNode >>
//                    int firstPrimOffset = orderedPrimitives.size();
//                    for (int i = start; i < end; ++i) {
//                        int primNum = primInfo[i].pid;
//                        orderedPrimitives.push_back(primitives[primNum]);
//                    }
//                    node->initLeafNode(box, firstPrimOffset, nPrimitives);
//                    return node;
//
//                }
//                node->initInteriorNode(recursiveBuild(primitives, primInfo, start, mid,
//                    totalNodes),
//                    recursiveBuild(primitives, primInfo, mid, end,
//                        totalNodes), dim);
//            }
//           
//        }
//    } return node;
//}
int BVH::toLinearTree(BVHBuildNode* root, int* offset) {
    if (root == nullptr) {
        return -1;
    }

    int nodeIdx = *offset;
    *offset += 1;
    int leftIdx = toLinearTree(root->leftChild, offset);
    int rightIdx = toLinearTree(root->rightChild, offset);
    nodes[nodeIdx].box = root->bound;
    if (leftIdx == -1 && rightIdx == -1) {
        nodes[nodeIdx].type = BVHNode::Type::Leaf;
        nodes[nodeIdx].startIndex = root->startIdx;
        nodes[nodeIdx].nPrimitives = root->nPrimitives;
    } else {
        nodes[nodeIdx].type = BVHNode::Type::NonLeaf;
        nodes[nodeIdx].leftChild = leftIdx;
        nodes[nodeIdx].rightChild = rightIdx;
    }

    return nodeIdx;
}

bool BVH::intersect(const Ray& ray, Interaction& isect) {
    bool hit = false;
    glm::vec3 invDir = glm::vec3(1.0f / ray.dir.x, 1.0f / ray.dir.y, 1.0f / ray.dir.z);
    int isDirNeg[3];
    isDirNeg[0] = ray.dir.x < 0 ? 1 : 0;
    isDirNeg[1] = ray.dir.y < 0 ? 1 : 0;
    isDirNeg[2] = ray.dir.z < 0 ? 1 : 0;
    int currentNodeIndex = 0;
    int toVisitOffset = 0;
    int nodesToVisit[128];
    BVHNode node;
    while (true) {
        node = nodes[currentNodeIndex];
        if (node.box.intersect(ray, invDir, isDirNeg)) {
            if (node.type == BVHNode::Type::Leaf) {
                int firstIndex = node.startIndex;
                int nPrimitives = node.nPrimitives;

                Primitive primitive;
                for (int i = 0; i < nPrimitives; ++i) {
                    primitive = orderedPrimitives[firstIndex + i];
                    if (intersectPrimitive(ray, primitive, isect)) {
                        hit = true;
                    }
                }

                if (toVisitOffset == 0) {
                    break;
                }

                currentNodeIndex = nodesToVisit[--toVisitOffset];
            } else {
                int leftChild = node.leftChild;
                int rightChild = node.rightChild;
                nodesToVisit[toVisitOffset++] = rightChild;
                currentNodeIndex = leftChild;
            }
        } else {
            if (toVisitOffset == 0) {
                break;
            }
            currentNodeIndex = nodesToVisit[--toVisitOffset];
        }
    }

    return hit;
}

AABB BVH::getAABB(const Primitive& prim) {
    if (prim.type == Primitive::Type::Sphere) {
        return getSphereAABB(*prim.sphere);
    }
    else {
        return getTriangleAABB(*prim.triangle);
    }
}

bool BVH::intersectPrimitive(const Ray& ray, const Primitive& primitive, Interaction& isect) {
    if (intersectSphere(ray, *primitive.sphere, isect)) {
        isect.primitive = primitive;
        return true;
    }

    return false;
}

AABB BVH::getTriangleAABB(const Triangle& triangle) {
    const auto& p1 = triangle.vertices[triangle.v[0]].position;
    const auto& p2 = triangle.vertices[triangle.v[1]].position;
    const auto& p3 = triangle.vertices[triangle.v[2]].position;
    return unionAABB(AABB(p1, p2), p3);
}

AABB BVH::getSphereAABB(const Sphere& sphere) {
    return AABB(sphere.position - glm::vec3(sphere.radius),
        sphere.position + glm::vec3(sphere.radius));
}

bool BVH::intersectSphere(const Ray& ray, const Sphere& sphere, Interaction& isect) {
    float a = glm::dot(ray.dir, ray.dir);
    float b = glm::dot(ray.dir, ray.o - sphere.position);
    float c = glm::dot(ray.o - sphere.position, ray.o - sphere.position) - sphere.radius * sphere.radius;
    float discriminant = b * b - a * c;

    if (discriminant >= 0) {
        float t1 = (-b - std::sqrt(discriminant)) / a;
        float t2 = (-b + std::sqrt(discriminant)) / a;

        if ((1e-3f <= t1 && t1 < ray.tMax) || (1e-3f <= t2 && t2 < ray.tMax)) {
            float t = (1e-3f <= t1 && t1 < ray.tMax) ? t1 : t2;
            ray.tMax = t;
            isect.hitPoint.position = ray.o + t * ray.dir;
            isect.hitPoint.normal = glm::normalize(isect.hitPoint.position - sphere.position);
            return true;
        }
    }

    return false;
}